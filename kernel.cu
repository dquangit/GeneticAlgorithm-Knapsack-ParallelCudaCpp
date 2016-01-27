#include "dq.h"
#define N 50
#define NewN 100

#define LifeN 500
#define numofthreads 512
int numofeles=0,capacity;

struct chromosome
{
	long long weight=0, value=0;
	bool chromo[100003];
};
chromosome chromoele[N],*cudaChromo,*cudaNewpopulation,newpopulation[NewN],res,x[2];
int weight[100001],value[100001],*devValue,*devWeight,*devnumeles;
bool myfun(chromosome a, chromosome b)
{
	if (a.value > b.value) return true;
	return false;
}
void bitonic_sort(chromosome *cudanewpopulation);
void loaddata(){
	gotoxy(1, 1);
	cout << "Loading Data from Disk to RAM";
	file.open("input.txt",ios::in);
	file >> capacity;
	while (file)
	 {
		file >> value[numofeles] >> weight[numofeles];
		numofeles++;
	} 
	
	file.close();
	numofeles--;

	gotoxy(1, 2); cout << "Loading Completed Successfully                   ";
	
}

__global__ void initpopulation(chromosome *cudaChromo,int seed,const int numofeles,int *devValue,int* devWeight)
{
	if (blockIdx.x < N){
		int idx = (threadIdx.x + blockIdx.x*blockDim.x);
		curandState state;
		curand_init(seed, idx, 1, &state);
		idx %= numofeles;
		bool tmp = curand(&state) % 2 == 1 ? true : false;
		cudaChromo[blockIdx.x].chromo[idx] = tmp;
	}
}
cudaError_t init(chromosome *cudaChromo, int seed, const int numele, int *devValue, int* devWeight);

__global__ void initOne(chromosome *cudaChromo, const int numele,int *devValue,int *devWeight)
{
	if (blockIdx.x < N){
		int idx = threadIdx.x + blockIdx.x*blockDim.x;
		idx %= numele;
		if (blockIdx.x == idx)
		{
			cudaChromo[blockIdx.x].chromo[idx] = true;
			cudaChromo[blockIdx.x].value = devValue[idx];
			cudaChromo[blockIdx.x].weight = devValue[idx];
		}
		else
			cudaChromo[blockIdx.x].chromo[idx] = false;
	}
}


__global__ void gan(chromosome *cudaChromo, chromosome* cudaNewpopulation,const int capacity)
{
	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < N) {
		for (int i = idx; i < NewN;i+=N)
		if (cudaNewpopulation[i].weight<=capacity&&cudaNewpopulation[i].value>cudaChromo[idx].value)
		cudaChromo[idx] = cudaNewpopulation[i];
	}
}

__global__ void evaluate(chromosome *cudaChromo,int *devValue,int *devWeight, int numele)
{
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	for (int i = 0; i < numele; i++){
		if (cudaChromo[idx].chromo[i])
		cudaChromo[idx].value += devValue[i];
		cudaChromo[idx].weight += (cudaChromo[idx].chromo[i] ? 1 : 0)*devWeight[i];
	}
	
}

cudaError_t evalute(chromosome *cudaChromo, int *devValue, int *devWeight, int numele);


__global__ void hybrid(chromosome *cudaChromo, chromosome *cudaNewpopulation, int seed1, const int numele, int *devValue, int *devWeight)
{

	int idx = threadIdx.x + blockIdx.x*blockDim.x;
	if (idx < NewN){
		curandState state;
		curand_init(seed1, idx, seed1, &state);
		int seed2 = curand(&state) % N;
		curand_init(seed1, idx, seed1, &state);
		int seed3 = curand(&state) % numele;
		cudaNewpopulation[idx] = cudaChromo[idx%N];
		
		if (idx <NewN-N){

			cudaNewpopulation[idx].value -= devValue[seed3] * (cudaNewpopulation[idx].chromo[seed3] ? 1 : 0);
			cudaNewpopulation[idx].weight -= devWeight[seed3] * (cudaNewpopulation[idx].chromo[seed3] ? 1 : 0);
			cudaNewpopulation[idx].chromo[seed3] = cudaChromo[seed2].chromo[seed3];
			cudaNewpopulation[idx].value += devValue[seed3] * (cudaNewpopulation[idx].chromo[seed3] ? 1 : 0);
			cudaNewpopulation[idx].weight += devWeight[seed3] * (cudaNewpopulation[idx].chromo[seed3] ? 1 : 0);
		}
		else{

			cudaNewpopulation[idx].chromo[seed3] = cudaNewpopulation[idx].chromo[seed3] ? false : true;
			//printf("\n%d\n", idx);
			cudaNewpopulation[idx].value += devValue[seed3] *(cudaNewpopulation[idx].chromo[seed3]? 1 : -1);
			cudaNewpopulation[idx].weight += devWeight[seed3] * (cudaNewpopulation[idx].chromo[seed3] ? 1 : -1);
		}

	}
}

cudaError_t hybridmutation(chromosome *cudaChromo, chromosome *cudaNewpopulation, int seed1, int numele, int *devValue, int *devWeight);


int main(void) {
	bool pass = true;
	cudaError_t cudastatus;
	cudastatus = cudaSetDevice(0);
	if (cudastatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	//cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1000*sizeof(chromosome));
	clock_t t;
	t = clock();
	srand(time(NULL));
	loaddata(); //load data from disk
	const int numele = numofeles;
	const int numofblocks = (numofeles / numofthreads) + (numofeles%numofthreads == 0 ? 0 : 1);
	// load data from RAM to GPU
	gotoxy(1, 4); cout << "Loading Data to GPU...";


	cudastatus=(cudaMalloc((void**)&devValue, numofeles *sizeof(int)));
	if (cudastatus != cudaSuccess) {
		cout << "OK1" << endl;
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	cudastatus=(cudaMalloc((void**)&devWeight, numofeles*sizeof(int)));
	if (cudastatus != cudaSuccess) {
		cout << "OK2" << endl;
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudastatus=(cudaMemcpy(devValue, value, numofeles *sizeof(int), cudaMemcpyHostToDevice));
	if (cudastatus != cudaSuccess) {
		cout << "OK3" << endl;
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudastatus=(cudaMemcpy(devWeight, weight, numofeles * sizeof(int), cudaMemcpyHostToDevice));
	if (cudastatus != cudaSuccess) {
		cout << "OK4" << endl;
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}
	
	/**************************/
	gotoxy(1, 5); cout << "Loading Completed Successfully                               ";
	gotoxy(1, 7); cout << "Caculating:      %";
	cudastatus=(cudaMalloc((void**)&cudaNewpopulation, NewN*sizeof(chromosome)));


	cudastatus=(cudaMalloc((void**)&cudaChromo, N *sizeof(chromosome)));
	cout << endl;

	cudastatus=init(cudaChromo, rand(), numele, devValue, devWeight);

	if (cudastatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudastatus));
		goto Error;
	}

	cudastatus = evalute(cudaChromo, devValue, devWeight, numele);

	/**************************************************/
	bool k = false;
	int d = 0;
		while (!k)
		{
			cudastatus = hybridmutation(cudaChromo, cudaNewpopulation, rand(), numele, devValue, devWeight);
			cudaMemcpy(newpopulation, cudaNewpopulation, NewN * sizeof(chromosome), cudaMemcpyDeviceToHost);
			for (int i = 0; i < NewN;i++)
			if (newpopulation[i].weight <= capacity)
			{
				k = true; break;
			}
			d++;
			if (d == 1)
				initOne<<<(N>numofblocks?N:numofblocks),numofthreads>>>(cudaChromo, numele, devValue, devWeight);
			if (d == 3)
			{
				cout << "Cannot find the answer" << endl;
				goto Error;
			}
			
		}
		/*********************************************************************/

		for (int C = 0; C < LifeN; C++){
		gotoxy(13, 7); cout <<(float) C / LifeN * 100 << " %       ";
		cudastatus = hybridmutation(cudaChromo, cudaNewpopulation, rand(), numele, devValue, devWeight);
		cudaDeviceSynchronize();
		gan << <N / numofthreads + (N%numofthreads == 0 ? 0 : 1), numofthreads >> >(cudaChromo, cudaNewpopulation,capacity);
		cudaDeviceSynchronize();
	}
		cudaMemcpy(chromoele, cudaChromo,N*sizeof(chromosome), cudaMemcpyDeviceToHost);
		sort(chromoele, chromoele + N, myfun);
		for (int i = 0; i<N; i++)
		{
			if (res.value < chromoele[i].value&&chromoele[i].weight<=capacity)
			{
				res = chromoele[i];
		
			}
		}
	gotoxy(13, 7); cout << "Completed                        " << endl;
	cout << " Value: "<<res.value << "\n Weight: " << res.weight << endl;
	cout << " Selection: ";
	if (numele > 20) cout << "Too long to display" << endl;
	else 
	for (int j = 0; j < numele; j++)
	{
		cout << res.chromo[j];
	}
	cout << endl;
	t = clock() - t;
	cout << " Time: " << ((float)t) / CLOCKS_PER_SEC << "s"<<endl;
Error:{
	cout << "Error: " << cudastatus << endl; }
	_getch();
	cudaFree(cudaChromo);
	cudaFree(devValue);
	cudaFree(devWeight);
	cudaFree(cudaNewpopulation);

}



cudaError_t init(chromosome *cudaChromo, int seed, const int numele, int *devValue, int* devWeight)
{
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	initpopulation << <(N>numofthreads ? N : numofthreads), numofthreads >> >(cudaChromo, rand(), numele, devValue, devWeight);
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
Error:{
	//cout << "Error: " << cudaStatus << endl;
}
	return cudaStatus;
}

cudaError_t evalute(chromosome *cudaChromo, int *devValue, int *devWeight, int numele)
{
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}
	evaluate << <1, N >> >(cudaChromo, devValue, devWeight, numele);
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
Error:{
	//cout << "Error: " << cudaStatus << endl;
}
	return cudaStatus;
}

cudaError_t hybridmutation(chromosome *cudaChromo, chromosome *cudaNewpopulation, int seed1, int numele, int *devValue, int *devWeight)
{
	cudaError_t cudaStatus;
	hybrid << <(NewN / numofthreads) + (NewN%numofthreads), numofthreads >> >(cudaChromo, cudaNewpopulation, rand(), numele, devValue, devWeight);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}
Error:{
//	cout << "Error: " << cudaStatus << endl;
}
	return cudaStatus;

}



__global__ void bitonic_sort_step(chromosome *cudanewpopulation, int j, int k)
{
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = threadIdx.x + blockDim.x * blockIdx.x;
	ixj = i^j;
	printf("                    %d                        \n", i);

	/* The threads with the lowest ids sort the array. */
	if ((ixj) > i) {
		if ((i&k) != 0) {
			/* Sort ascending */
			if (cudanewpopulation[i].value < cudanewpopulation[ixj].value) {
				/* exchange(i,ixj); */
				chromosome temp = cudanewpopulation[i];
				cudanewpopulation[i] = cudanewpopulation[ixj];
				cudanewpopulation[ixj] = temp;

			}
		}
	}

}


/**
76  * Inplace bitonic sort using CUDA.
77  */
void bitonic_sort(chromosome *cudanewpopulation)
{
	int j, k;
	/* Major step */
	for (k = 2; k <= NewN; k <<= 1) {
		/* Minor step */
		for (j = k >> 1; j>0; j = j >> 1) {
			bitonic_sort_step << <(NewN / numofthreads + (NewN%numofthreads == 0) ? 0 : 1), numofthreads >> >(cudanewpopulation, j, k);
			cudaDeviceSynchronize();
		}

	}

}




