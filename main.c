#include "ocl_boiler.h"

int error(const char *msg) {
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

int verify(int *vsum, int nels) {
	int ret = 0;
	for (int i = 0 ; i < nels; ++i) {
		if (vsum[i] != nels) {
			fprintf(stderr, "%d: %d != %d\n",
				i, vsum[i], nels);
			ret = 1;
			break;
		}
	}
	return ret;
}

int lws_cli;
int vec_kernel;

size_t preferred_wg_init;
cl_event init(cl_command_queue que,
	cl_kernel init_k,
	cl_mem d_v1, cl_mem d_v2, cl_int nels)
{
	size_t lws[] = { lws_cli };
	/* se il local work size è stato specificato, arrotondiamo il
	 * global work size al multiplo successivo del lws, altrimenti,
	 * lo arrotondiamo al multiplo successivo della base preferita
	 * dalla piattaforma */
	size_t gws[] = {
		round_mul_up(nels/vec_kernel,
			lws_cli ? lws[0] : preferred_wg_init)
	};
	cl_event init_evt;
	cl_int err;

	printf("init gws: %d | %d | %zu => %zu\n",
		nels, lws_cli, preferred_wg_init, gws[0]);

	err = clSetKernelArg(init_k, 0, sizeof(d_v1), &d_v1);
	ocl_check(err, "set init arg 0");
	err = clSetKernelArg(init_k, 1, sizeof(d_v2), &d_v2);
	ocl_check(err, "set init arg 1");
	err = clSetKernelArg(init_k, 2, sizeof(nels), &nels);
	ocl_check(err, "set init arg 2");

	err = clEnqueueNDRangeKernel(que, init_k,
		1, NULL, gws, (lws_cli ? lws : NULL), /* griglia di lancio */
		0, NULL, /* waiting list */
		&init_evt);
	ocl_check(err, "enqueue kernel init");

	return init_evt;
}

size_t preferred_wg_sum;
cl_event sum(cl_command_queue que,
	cl_kernel sum_k,
	cl_mem d_vsum, cl_mem d_v1, cl_mem d_v2, cl_int nels,
	cl_event init_evt)
{
	size_t lws[] = { lws_cli };
	/* se il local work size è stato specificato, arrotondiamo il
	 * global work size al multiplo successivo del lws, altrimenti,
	 * lo arrotondiamo al multiplo successivo della base preferita
	 * dalla piattaforma */
	size_t gws[] = {
		round_mul_up(nels/vec_kernel,
			lws_cli ? lws[0] : preferred_wg_sum)
	};
	cl_event sum_evt;
	cl_int err;

	printf("sum gws: %d | %d | %zu => %zu\n",
		nels, lws_cli, preferred_wg_sum, gws[0]);

	err = clSetKernelArg(sum_k, 0, sizeof(d_vsum), &d_vsum);
	ocl_check(err, "set sum arg 0");
	err = clSetKernelArg(sum_k, 1, sizeof(d_v1), &d_v1);
	ocl_check(err, "set sum arg 1");
	err = clSetKernelArg(sum_k, 2, sizeof(d_v2), &d_v2);
	ocl_check(err, "set sum arg 2");
	err = clSetKernelArg(sum_k, 3, sizeof(nels), &nels);
	ocl_check(err, "set sum arg 3");

	cl_event wait_list[] = { init_evt };
	err = clEnqueueNDRangeKernel(que, sum_k,
		1, NULL, gws, (lws_cli ? lws : NULL), /* griglia di lancio */
		1, wait_list, /* waiting list */
		&sum_evt);
	ocl_check(err, "enqueue kernel sum");

	return sum_evt;
}


int main(int argc, char *argv[])
{
	if (argc < 2)
		error("sintassi: vecsum_ocl_vec numels [lws] [vec]");

	int nels = atoi(argv[1]); /* numero di elementi */
	if (nels <= 0)
		error("il numero di elementi deve essere positivo");

	if (argc >= 3)
		lws_cli = atoi(argv[2]); /* local work size */
	if (lws_cli < 0)
		error("il local work size deve essere non negativo");

	if (argc >= 4)
		vec_kernel = atoi(argv[3]); /* kernel vettoriale */
	else
		vec_kernel = 1;
	if (vec_kernel != 1 && vec_kernel != 4 && vec_kernel != 16)
		error("solo kernel scalari o vettoriali a 4 e 16 sono supportati");

	const size_t memsize = sizeof(int)*nels;

	/* Hic sunt leones */

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("kernels.ocl", ctx, d);

	cl_int err;

	/* Extract kernels */

	/* per i kernel vettoriali, nels deve essere multiplo
	 * della larghezza vettoriale del kernel */
	if (vec_kernel == 16 && (nels & 15))
		error("vec_kernel 16, nels non multiplo di 16");
	if (vec_kernel == 4 && (nels & 3))
		error("vec_kernel 4, nels non multiplo di 4");

	cl_kernel init_k = clCreateKernel(prog,
		((vec_kernel == 16) ? "init16" :
		 ((vec_kernel == 4) ? "init4" : "init")),
		&err);
	ocl_check(err, "create kernel init");
	cl_kernel sum_k = clCreateKernel(prog,
		((vec_kernel == 16) ? "sum16" :
		 ((vec_kernel == 4) ? "sum4" : "sum")),
		&err);
	ocl_check(err, "create kernel sum");

	err = clGetKernelWorkGroupInfo(init_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(preferred_wg_init), &preferred_wg_init, NULL);
	err = clGetKernelWorkGroupInfo(sum_k, d,
		CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
		sizeof(preferred_wg_sum), &preferred_wg_sum, NULL);

	cl_mem d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
		memsize, NULL, &err);
	ocl_check(err, "create buffer v1");
	cl_mem d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
		memsize, NULL, &err);
	ocl_check(err, "create buffer v2");
	cl_mem d_vsum = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
		memsize, NULL, &err);
	ocl_check(err, "create buffer vsum");

	cl_event init_evt = init(que, init_k, d_v1, d_v2, nels);
	cl_event sum_evt = sum(que, sum_k,
		d_vsum, d_v1, d_v2, nels,
		init_evt);

	int *vsum = malloc(memsize);
	if (!vsum)
		error("alloc vsum");

	cl_event copy_evt;
	err = clEnqueueReadBuffer(que, d_vsum, CL_TRUE,
		0, memsize, vsum,
		1, &sum_evt, &copy_evt);
	ocl_check(err, "read buffer vsum");

	printf("init time:\t%gms\t%gGB/s\n", runtime_ms(init_evt),
		(2.0*memsize)/runtime_ns(init_evt));
	printf("sum time:\t%gms\t%gGB/s\n", runtime_ms(sum_evt),
		(3.0*memsize)/runtime_ns(sum_evt));
	printf("copy time:\t%gms\t%gGB/s\n", runtime_ms(copy_evt),
		(2.0*memsize)/runtime_ns(copy_evt));

	verify(vsum, nels);

	return 0;
}
