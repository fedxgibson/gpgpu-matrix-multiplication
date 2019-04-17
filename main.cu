#include "spmvJDS.cuh"

bool compare(float* vector1, float* vector2, int largo){

	for(int i = 0; i < largo; i++){
		if(vector1[i] != vector2[i]){
			return printf("Error %i CPU %f GPU %f \n",i,vector1[i],vector2[i]);
		}
	}

	return true;
}

void matrizPrueba(int M, int N, float* &val, int* &col_idx,int* &col_start,int* &perm_vector, int &num_jdiags, float* &x){
	val = new float[M*N];
	col_idx	= new int[M*N];
	col_start	= new int[N+1];
	perm_vector	= new int[M];
	num_jdiags = N;
	x = new float[N];

	for(int i = 0; i < N; i++){
		col_start[i] = i*M;
		x[i] = 1;
		for(int j = 0; j < M; j++){			
			val[i*M+j] = 1;
			col_idx[i*M+j] = i;
			if(i == 0){
				perm_vector[j] = j;
			}
		}
	}
	col_start[N] = M*N;
}

void randomJDS(int M,int N, int offsetBottom, float load, float* &val,int* &col_idx,int* &col_start,int* &perm_vector, int &num_jdiags){

	int cantElem = std::floor((N * load / 100.0f) * M);
	int topeFila = M - offsetBottom;
	int idColumna = 0;	
	int idFila = 0;
	int ind = 0;
	
	//Inicio
	val = (float*)malloc(sizeof(float)*cantElem);
	col_idx = (int*)malloc(sizeof(int)*cantElem);
	col_start = (int*)malloc(sizeof(int)*(N+1));
	perm_vector = (int*)malloc(sizeof(int)*M);		
	col_start[0] = 0;
	num_jdiags = 1;

	while(cantElem > 0 && idColumna < N){		
		//Agrego valor
		val[ind] = rand() % 10 + 1;
		col_idx[ind] = idColumna;
		cantElem--;
		ind++;

		//Condicion de cambio de columna
		idFila++;
		if(idFila >= topeFila && cantElem > 0){			
			if(rand() % 10000 < 10){
				double div = cantElem;
				div = div / (N - idColumna - 1);
				double fDiv = std::floor(div);
				if(div > fDiv){
					div = fDiv+1;
				}else{
					div = fDiv;
				}
				int rango = (idFila - div - 1);
				if(rango > 0){
					topeFila = rand() % rango + div;
				}
			}
			idFila = 0;
			idColumna++;
			col_start[idColumna] = ind;
			if(idColumna < N)
				num_jdiags++;
		}
	}
	
	col_start[num_jdiags] = ind;
	
	for (int i = 0; i < M; i++){
		perm_vector[i] = M - i - 1;
	}
}

void generateX(int N, float* &x){
	x = (float*)malloc(sizeof(float)*N);
	for (int i = 0; i < N; i++){
		x[i] = rand() % 10 + 1;
	}
}

int main()
{
	int M = 10240;
	int N = 10240;
	int cantPruebas = 200;
	float carga = 1;
	int chunkSize = 256;
	float* y_cpu = 0;
	float* y_gpu = 0;
	float* val = 0; 
	float* x = 0;
	int* col_idx = 0; 
	int* col_start = 0;
	int* perm_vector = 0;
	int num_jdiags = 0;
	
	srand (time(NULL));

	printf("Prueba %i Filas %i Columnas %f Carga %i Iteraciones\n",M,N,carga,cantPruebas);

	for(int i = 0; i < cantPruebas; i++){
		//Valores de prueba
		randomJDS(M,N,0,carga,val, col_idx, col_start, perm_vector,num_jdiags);
		generateX(N,x);

		printf("jdiags: %i\t",num_jdiags);
				
		//CPU
		y_cpu = spmvJDS_CPU(val, col_idx, col_start, perm_vector, x, M, N,num_jdiags);

		//Kernel 0
		y_gpu = spmvJDSv0_hst(val, col_idx, col_start, perm_vector, x, M, N,num_jdiags,chunkSize);
		if(!compare(y_cpu,y_gpu,M)){
			printf("Error K0\n");
		}
		free(y_gpu);

		//Kernel 1
		y_gpu = spmvJDSv1_hst(val, col_idx, col_start, perm_vector, x, M, N,num_jdiags,chunkSize);
		if(!compare(y_cpu,y_gpu,M)){
			printf("Error K1\n");
		}
		free(y_gpu);
		
		//Kernel 2
		y_gpu = spmvJDSv2_hst(val, col_idx, col_start, perm_vector, x, M, N,num_jdiags,chunkSize);
		if(!compare(y_cpu,y_gpu,M)){
			printf("Error K2\n");
		}
		free(y_gpu);

		//Kernel 3
		y_gpu = spmvJDSv3_hst(val, col_idx, col_start, perm_vector, x, M, N,num_jdiags,chunkSize);
		if(!compare(y_cpu,y_gpu,M)){
			printf("Error K3\n");
		}
		free(y_gpu);
		
		free(y_cpu);
		free(val); 
		free(x);
		free(col_idx); 
		free(col_start);
		free(perm_vector);
		num_jdiags = 0;
		printf("\n");
	}
	scanf("%d");

	return 0;
}
