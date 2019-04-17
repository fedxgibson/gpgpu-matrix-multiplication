#include "spmvJDS.cuh"


//Helper Functions

__int64 ctr1 = 0, ctr2 = 0, freq = 0;

void clockStart(){
	QueryPerformanceCounter((LARGE_INTEGER *)&ctr1);
}

void clockStop(const char * str){

	QueryPerformanceCounter((LARGE_INTEGER *)&ctr2);
	QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	//printf("%s : %fs\t", str, (ctr2 - ctr1) * 1.0 / freq);
	printf("%f\t", (ctr2 - ctr1) * 1.0 / freq);
}

//Device Functions
__constant__ int dev_num_jdiag;
__global__ void spmvJDSv0_dev(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, float* y){
	float value = 0;
	int idColumna = blockIdx.y;
	int idFila = blockIdx.x * blockDim.x + threadIdx.x;
	int j = col_start[idColumna];

	if(idColumna < dev_num_jdiag && (col_start[idColumna+1] - j) > idFila){				
		value = val[j+idFila] * x[col_idx[j+idFila]];
		atomicAdd(&y[perm_vector[idFila]], value);
	}
}

__global__ void spmvJDSv1_dev(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, float* y){
	float value = 0;
	int idFila = blockIdx.x*blockDim.x + threadIdx.x;
	int i = 0;
	int j, j2;
	
	if(i < dev_num_jdiag){
		j = col_start[i];
		j2 = col_start[i+1];
		if((j2-j) > idFila){
			do{
				value += val[j+idFila] * x[col_idx[j+idFila]];
				i++;
				j = j2;
				j2 = col_start[i+1]; 
			}
			while(i < dev_num_jdiag && (j2-j) > idFila);

			y[perm_vector[idFila]] = value;
		}		
	}
}

__global__ void spmvJDSv2_dev(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, float* y){
	float value = 0;
	int idFila = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.x;	
	int j, j2;
	int tope = min(i + blockDim.x,dev_num_jdiag);
	
	if(i < tope){
		j = col_start[i];
		j2 = col_start[i+1];
		if((j2-j) > idFila){
			do{
				value += val[j+idFila] * x[col_idx[j+idFila]];
				i++;
				j = j2;
				j2 = col_start[i+1]; 
			}
			while(i < tope && (j2-j) > idFila);
		}

		atomicAdd(&y[perm_vector[idFila]], value);
	}
}

__global__ void spmvJDSv3_dev(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, float* y){
	extern __shared__ int col_start_shared[];
	int* col_start_vector = col_start;
	float value = 0;
	int idFila = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.x;	
	int j, j2;
	int tope = min(i + blockDim.x, dev_num_jdiag);

	//Copiamos colstart a shared
	if(blockIdx.y * (blockDim.x + 1) * sizeof(int) < SHARED_SIZE && i < tope){
		col_start_shared[threadIdx.x] = col_start[i + threadIdx.x];
		if(threadIdx.x == (tope - i - 1)){
			col_start_shared[threadIdx.x + 1] = col_start[tope];
		}
		
		__syncthreads();

		i = 0;
		tope = tope - blockIdx.y * blockDim.x;		
		col_start_vector = col_start_shared;
	}
	
	if(i < tope){
		j = col_start_vector[i];
		j2 = col_start_vector[i+1];
		if((j2-j) > idFila){
			do{
				value += val[j+idFila] * x[col_idx[j+idFila]];
				i++;
				j = j2;
				j2 = col_start_vector[i+1]; 
			}
			while(i < tope && (j2-j) > idFila);
		}

		atomicAdd(&y[perm_vector[idFila]], value);
	}
}

//Host Functions
float* spmvJDS_CPU(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags){
	int i, j, k;
	float* y = (float*)malloc(sizeof(float)*M);
	float* y_perm = (float*)malloc(sizeof(float)*M);
	for(int i = 0; i < M; i++){
		y_perm[i] = 0;
	}
	
	clockStart();
	for (i = 0; i < num_jdiags; i++){
		k = 0;
		for (j = col_start[i]; j  < col_start[i + 1]; j++){
			y_perm[k] += val[j] * x[col_idx[j]];
			k++;
		}
	}

	for (i = 0; i < M; i++){
		y[perm_vector[i]] = y_perm[i];
	}

	clockStop("CPU");

	free(y_perm);
	return y;

}

float* spmvJDSv0_hst(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags, int chunkSize){
	// arrays en el device
	float* val_dev = 0;
	int* col_idx_dev = 0;
	int* col_start_dev = 0;
	int* perm_vector_dev = 0;
	float* x_dev = 0;
	float* y_dev = 0;
	float* y = 0;
	int xBloques;

	y = (float*)malloc(sizeof(float)*M);

	//Configuracion de grilla
	xBloques = col_start[1] / chunkSize;
	if(col_start[1] % chunkSize > 0){
		xBloques++;
	}
	dim3 tamGrid = dim3(xBloques,num_jdiags);
	dim3 tamBlock(chunkSize, 1); 

	//Copiamos constante
	cudaMemcpyToSymbol(dev_num_jdiag,&num_jdiags,sizeof(int));
		
	//memoria para arrays en dispositivo
	cudaMalloc(&val_dev, sizeof(float) * col_start[num_jdiags]);
	cudaMalloc(&col_idx_dev, sizeof(int) * col_start[num_jdiags]);
	cudaMalloc(&col_start_dev, sizeof(int) * (num_jdiags+1));
	cudaMalloc(&perm_vector_dev, sizeof(int) * M);
	cudaMalloc(&x_dev, sizeof(float) * N);
	cudaMalloc(&y_dev, sizeof(float) * M);
		
	//Copiamos datos a device
	cudaMemcpy(val_dev, val, sizeof(float) * col_start[num_jdiags], cudaMemcpyHostToDevice);
	cudaMemcpy(col_idx_dev, col_idx, sizeof(int) * col_start[num_jdiags], cudaMemcpyHostToDevice);
	cudaMemcpy(col_start_dev, col_start, sizeof(int) * (num_jdiags+1), cudaMemcpyHostToDevice);
	cudaMemcpy(perm_vector_dev, perm_vector, sizeof(int) * M, cudaMemcpyHostToDevice);
	cudaMemcpy(x_dev, x, sizeof(float) * N, cudaMemcpyHostToDevice);
	
	//Inicializamos salida
	cudaMemset(y_dev, 0, sizeof(float)*M);

	//Bloqueo hasta que el dispositivo se ha completado todas las tareas solicitadas	
	cudaDeviceSynchronize();
	
	clockStart();
	spmvJDSv0_dev<<<tamGrid, tamBlock>>>(val_dev, col_idx_dev, col_start_dev, perm_vector_dev, x_dev, y_dev);
	cudaDeviceSynchronize();
	clockStop("Kernel 0");
	
	// copiar array de salida desde el dispositivo...
	cudaMemcpy(y, y_dev, sizeof(float)*M , cudaMemcpyDeviceToHost);  

	//Liberamos recursos
	cudaFree(val_dev);
	cudaFree(col_idx_dev);
	cudaFree(col_start_dev);
	cudaFree(perm_vector_dev);
	cudaFree(x_dev);
	cudaFree(y_dev);

	return y;
}

float* spmvJDSv1_hst(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags, int chunkSize){
	// arrays en el device
	float* val_dev = 0;
	int* col_idx_dev = 0;
	int* col_start_dev = 0;
	int* perm_vector_dev = 0;
	float* x_dev = 0;
	float* y_dev = 0;
	float* y = 0;
	int xBloques;

	y = (float*)malloc(sizeof(float)*M);

	//Configuracion de grilla
	xBloques = col_start[1] / chunkSize;
	if(col_start[1] % chunkSize > 0){
		xBloques++;
	}
	dim3 tamGrid = dim3(xBloques,1);
	dim3 tamBlock(chunkSize, 1); 
	
	//Copiamos constante
	cudaMemcpyToSymbol(dev_num_jdiag,&num_jdiags,sizeof(int));
	
	//memoria para arrays en dispositivo
	cudaMalloc(&val_dev, sizeof(float) * col_start[num_jdiags]);
	cudaMalloc(&col_idx_dev, sizeof(int) * col_start[num_jdiags]);
	cudaMalloc(&col_start_dev, sizeof(int) * (num_jdiags+1));
	cudaMalloc(&perm_vector_dev, sizeof(int) * M);
	cudaMalloc(&x_dev, sizeof(float) * N);
	cudaMalloc(&y_dev, sizeof(float) * M);

	//Copiamos datos a device
	cudaMemcpy(val_dev, val, sizeof(float) * col_start[num_jdiags], cudaMemcpyHostToDevice);
	cudaMemcpy(col_idx_dev, col_idx, sizeof(int) * col_start[num_jdiags], cudaMemcpyHostToDevice);
	cudaMemcpy(col_start_dev, col_start, sizeof(int) * (num_jdiags+1), cudaMemcpyHostToDevice);
	cudaMemcpy(perm_vector_dev, perm_vector, sizeof(int) * M, cudaMemcpyHostToDevice);
	cudaMemcpy(x_dev, x, sizeof(float) * N, cudaMemcpyHostToDevice);

	//Inicializamos salida
	cudaMemset(y_dev, 0, sizeof(float)*M);

	//Bloqueo hasta que el dispositivo se ha completado todas las tareas solicitadas	
	cudaDeviceSynchronize();
	
	clockStart();
	spmvJDSv1_dev<<<tamGrid, tamBlock>>>(val_dev, col_idx_dev, col_start_dev, perm_vector_dev, x_dev, y_dev);
	cudaDeviceSynchronize();
	clockStop("Kernel 1");
	
	// copiar array de salida desde el dispositivo...
	cudaMemcpy(y, y_dev, sizeof(float)*M , cudaMemcpyDeviceToHost);  

	//Liberamos recursos
	cudaFree(val_dev);
	cudaFree(col_idx_dev);
	cudaFree(col_start_dev);
	cudaFree(perm_vector_dev);
	cudaFree(x_dev);
	cudaFree(y_dev);

	return y;
}

float* spmvJDSv2_hst(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags, int chunkSize){
	// arrays en el device
	float* val_dev = 0;
	int* col_idx_dev = 0;
	int* col_start_dev = 0;
	int* perm_vector_dev = 0;
	float* x_dev = 0;
	float* y_dev = 0;
	float* y = 0;
	int xBloques;
	int yBloques;

	y = (float*)malloc(sizeof(float)*M);

	//Configuracion de grilla
	xBloques = col_start[1] / chunkSize;
	if(col_start[1] % chunkSize > 0){
		xBloques++;
	}
	yBloques = num_jdiags / chunkSize;
	if(num_jdiags % chunkSize > 0){
		yBloques++;
	}
	dim3 tamGrid = dim3(xBloques,yBloques);
	dim3 tamBlock(chunkSize, 1); 
	
	//Copiamos constante
	cudaMemcpyToSymbol(dev_num_jdiag,&num_jdiags,sizeof(int));
	
	//memoria para arrays en dispositivo
	cudaMalloc(&val_dev, sizeof(float) * col_start[num_jdiags]);
	cudaMalloc(&col_idx_dev, sizeof(int) * col_start[num_jdiags]);
	cudaMalloc(&col_start_dev, sizeof(int) * (num_jdiags+1));
	cudaMalloc(&perm_vector_dev, sizeof(int) * M);
	cudaMalloc(&x_dev, sizeof(float) * N);
	cudaMalloc(&y_dev, sizeof(float) * M);

	//Copiamos datos a device
	cudaMemcpy(val_dev, val, sizeof(float) * col_start[num_jdiags], cudaMemcpyHostToDevice);
	cudaMemcpy(col_idx_dev, col_idx, sizeof(int) * col_start[num_jdiags], cudaMemcpyHostToDevice);
	cudaMemcpy(col_start_dev, col_start, sizeof(int) * (num_jdiags+1), cudaMemcpyHostToDevice);
	cudaMemcpy(perm_vector_dev, perm_vector, sizeof(int) * M, cudaMemcpyHostToDevice);
	cudaMemcpy(x_dev, x, sizeof(float) * N, cudaMemcpyHostToDevice);

	//Inicializamos salida
	cudaMemset(y_dev, 0, sizeof(float)*M);

	//Bloqueo hasta que el dispositivo se ha completado todas las tareas solicitadas	
	cudaDeviceSynchronize();
	
	clockStart();
	spmvJDSv2_dev<<<tamGrid, tamBlock>>>(val_dev, col_idx_dev, col_start_dev, perm_vector_dev, x_dev, y_dev);
	cudaDeviceSynchronize();
	clockStop("Kernel 2");
	
	// copiar array de salida desde el dispositivo...
	cudaMemcpy(y, y_dev, sizeof(float)*M , cudaMemcpyDeviceToHost);  

	//Liberamos recursos
	cudaFree(val_dev);
	cudaFree(col_idx_dev);
	cudaFree(col_start_dev);
	cudaFree(perm_vector_dev);
	cudaFree(x_dev);
	cudaFree(y_dev);

	return y;
}

float* spmvJDSv3_hst(float* val, int* col_idx, int* col_start, int* perm_vector, float* x, int M, int N, int num_jdiags, int chunkSize){
	// arrays en el device
	float* val_dev = 0;
	int* col_idx_dev = 0;
	int* col_start_dev = 0;
	int* perm_vector_dev = 0;
	float* x_dev = 0;
	float* y_dev = 0;
	float* y = 0;
	int xBloques;
	int yBloques;

	y = (float*)malloc(sizeof(float)*M);

	//Configuracion de grilla
	xBloques = col_start[1] / chunkSize;
	if(col_start[1] % chunkSize > 0){
		xBloques++;
	}
	yBloques = num_jdiags / chunkSize;
	if(num_jdiags % chunkSize > 0){
		yBloques++;
	}
	dim3 tamGrid = dim3(xBloques,yBloques);
	dim3 tamBlock(chunkSize, 1); 
	
	//Indicamos que necesitamos shared mas grande que L1
	cudaFuncSetCacheConfig(spmvJDSv3_dev,cudaFuncCachePreferShared);

	//Copiamos constante
	cudaMemcpyToSymbol(dev_num_jdiag,&num_jdiags,sizeof(int));
	
	//memoria para arrays en dispositivo
	cudaMalloc(&val_dev, sizeof(float) * col_start[num_jdiags]);
	cudaMalloc(&col_idx_dev, sizeof(int) * col_start[num_jdiags]);
	cudaMalloc(&col_start_dev, sizeof(int) * (num_jdiags+1));
	cudaMalloc(&perm_vector_dev, sizeof(int) * M);
	cudaMalloc(&x_dev, sizeof(float) * N);
	cudaMalloc(&y_dev, sizeof(float) * M);

	//Copiamos datos a device
	cudaMemcpy(val_dev, val, sizeof(float) * col_start[num_jdiags], cudaMemcpyHostToDevice);
	cudaMemcpy(col_idx_dev, col_idx, sizeof(int) * col_start[num_jdiags], cudaMemcpyHostToDevice);
	cudaMemcpy(col_start_dev, col_start, sizeof(int) * (num_jdiags+1), cudaMemcpyHostToDevice);
	cudaMemcpy(perm_vector_dev, perm_vector, sizeof(int) * M, cudaMemcpyHostToDevice);
	cudaMemcpy(x_dev, x, sizeof(float) * N, cudaMemcpyHostToDevice);

	//Inicializamos salida
	cudaMemset(y_dev, 0, sizeof(float)*M);

	//Bloqueo hasta que el dispositivo se ha completado todas las tareas solicitadas	
	cudaDeviceSynchronize();

	clockStart();
	spmvJDSv3_dev<<<tamGrid, tamBlock, min(sizeof(int)*yBloques*(chunkSize+1),SHARED_SIZE)>>>(val_dev, col_idx_dev, col_start_dev, perm_vector_dev, x_dev, y_dev);
	cudaDeviceSynchronize();
	clockStop("Kernel 3");
	
	// copiar array de salida desde el dispositivo...
	cudaMemcpy(y, y_dev, sizeof(float)*M , cudaMemcpyDeviceToHost);  

	//Liberamos recursos
	cudaFree(val_dev);
	cudaFree(col_idx_dev);
	cudaFree(col_start_dev);
	cudaFree(perm_vector_dev);
	cudaFree(x_dev);
	cudaFree(y_dev);

	return y;
}
