/*
prodmat.c
Lê duas matrizes geradas pelo MATLAB e calcula o produto e a norma 2 do mesmo em diversas precisões.
Uso:
	prodmat
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#define bufsize 10000			// para leitura dos dados em arquivo
#define POSROWNBR 7				// posição do número de linhas no arquivo
#define POSCOLNBR 10			// posição do número de colunas no arquivo


double fgetval(char ** pbuffer);
double * dmmult(double * pA, int nrowA, int ncolA, double * pB, int nrowB, int ncolB);
double dmnorm2(double * pmat, int nrow, int ncol);
float fmnorm2(float * pmat, int nrow, int ncol);
long double ldmnorm2(long double * pmat, int nrow, int ncol);
float * fmcopy(double * psrc, int nrows, int ncols);
float * fmmult(float * pA, int nrowA, int ncolA, float * pBf, int nrowB, int ncolB);
long double * ldmcopy(double * psrc, int nrows, int ncols);
long double * ldmmult(long double * pA, int nrowA, int ncolA, long double * pB, int nrowB, int ncolB);
double * lermat(const char * fname, int * nrows, int * ncolA);
int main(int argc, const char * argv[]);

static int debuglevel_ = 0;

int main(int argc, const char * argv[]) {
	// Lê as matrizes de entrada
	float * pAf, * pBf, * pCf, fnorm;
	double * pAd, * pBd, * pCd, dnorm;
	long double * pAld, * pBld, * pCld, ldnorm;
	// __float128 * pAq, * pBq, * pCq, qnorm;
	int nrowA, ncolA, nrowB, ncolB, nrowC, ncolC;
	pAd = lermat("MatA", & nrowA, & ncolA);
	if (pAd == NULL) {
		return 1;
		}
	pBd = lermat("MatB", & nrowB, & ncolB );
	if (pBd == NULL) {
		return 1;
		}
	// Verifica se podem ser multiplicadas
	if (ncolA != nrowB) {
		printf("As matrizes não podem ser multiplicadas, porque as dimensões são incompatíveis: (%d,%d) x (%d,%d)", nrowA, ncolA, nrowB, ncolB);
		return 2;
		}
	// Cria versões em diversas precisões
	pAf = fmcopy(pAd, nrowA, ncolA);
	pBf = fmcopy(pBd, nrowB, ncolB);
	pAld = ldmcopy(pAd, nrowA, ncolA);
	pBld = ldmcopy(pBd, nrowB, ncolB);
	// Multiplica as matrizes
	pCf = fmmult(pAf, nrowA, ncolA, pBf, nrowB, ncolB);
	pCd = dmmult(pAd, nrowA, ncolA, pBd, nrowB, ncolB);
	pCld = ldmmult(pAld, nrowA, ncolA, pBld, nrowB, ncolB);
	// Calcula a norma 2 dos resultados
	fnorm = fmnorm2(pCf, nrowA, ncolB);
	dnorm = dmnorm2(pCd, nrowA, ncolB);
	ldnorm = ldmnorm2(pCld, nrowA, ncolB);
	printf("Norma 2 do resultado: 32 bits = %f, 64 bits = %f, 80 bits = %f", fnorm, dnorm, (double) ldnorm);	
	return 0;
	}


// Funções para cópia das matrizes em diversas precisões	
float * fmcopy(double * psrc, int nrows, int ncols) {
// Retorna uma cópia em precisão simples (32 bits) da matriz 'psrc'
	float * pdst, * result;
	int size = nrows * ncols;
	result = pdst = (float *) malloc (size * sizeof(float));
	if (pdst == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		return NULL;
		}
	while (size -- > 0) {
		* pdst ++ = * psrc ++;
		}
	return result;
	}
	
long double * ldmcopy(double * psrc, int nrows, int ncols) {
// Retorna uma cópia em precisão estendida (80 bits) da matriz 'psrc'
	long double * pdst, * result;
	int size = nrows * ncols;
	result = pdst = (long double *) malloc (size * sizeof(long double));
	if (pdst == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		return NULL;
		}
	while (size -- > 0) {
		double dvalor;
		long double ldvalor;
		dvalor = * psrc ++;
		ldvalor = dvalor;
		* pdst ++ = ldvalor;
		if (debuglevel_ >= 3) {
			printf(" %f -> %lf", dvalor, ldvalor);
			}
		}
	return result;
	}

// Funções para multimplicação das matrizes em diversas precisões
float * fmmult(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB) {
	int sizeC = nrowA * ncolB;
	float * pvals , * result;
	result = pvals = (float *) calloc(sizeC, sizeof(float));
	if (pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrowA, ncolB);
		return NULL;
		}
	for (int i = 0; i < nrowA; ++ i) {
		for (int j = 0; j < ncolB; ++ j) {
			for (int k = 0; k < ncolA; ++ k) {
				pvals[i * ncolB + j] += pA[i * ncolA + k] * pB[k * ncolB + j];
				}
			}
		}
	if (debuglevel_ >= 2) {
		printf("A x B = C (float) \n A \n");
		for (int i = 0; i < nrowA; ++ i) {
			for (int j = 0; j < ncolA; ++ j) {
				printf(" %f ", pA[i * ncolA + j]);
				}
			printf("\n");
			}
		printf("B \n");
		for (int i = 0; i < nrowB; ++ i) {
			for (int j = 0; j < ncolB; ++ j) {
				printf(" %f ", pB[i * ncolB + j]);
				}
			printf("\n");
			}
		printf("C \n");
		for (int i = 0; i < nrowA; ++ i) {
			for (int j = 0; j < ncolB; ++ j) {
				printf(" %f ", result[i * ncolB + j]);
				}
			printf("\n");
			}
		}
	return result;
	}
	
double * dmmult(double * pA, int nrowA, int ncolA, double * pB, int nrowB, int ncolB) {
	int sizeC = nrowA * ncolB;
	double * pvals , * result;
	result = pvals = (double *) calloc(sizeC, sizeof(double));
	if (pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrowA, ncolB);
		return NULL;
		}
	for (int i = 0; i < nrowA; ++ i) {
		for (int j = 0; j < ncolB; ++ j) {
			for (int k = 0; k < ncolA; ++ k) {
				pvals[i * ncolB + j] += pA[i * ncolA + k] * pB[k * ncolB + j];
				}
			}
		}
	if (debuglevel_ >= 2) {
		printf("A x B = C (double) \n A \n");
		for (int i = 0; i < nrowA; ++ i) {
			for (int j = 0; j < ncolA; ++ j) {
				printf(" %f ", pA[i * ncolA + j]);
				}
			printf("\n");
			}
		printf("B \n");
		for (int i = 0; i < nrowB; ++ i) {
			for (int j = 0; j < ncolB; ++ j) {
				printf(" %f ", pB[i * ncolB + j]);
				}
			printf("\n");
			}
		printf("C \n");
		for (int i = 0; i < nrowA; ++ i) {
			for (int j = 0; j < ncolB; ++ j) {
				printf(" %f ", result[i * ncolB + j]);
				}
			printf("\n");
			}
		}
	return result;
	}

long double * ldmmult(long double * pA, int nrowA, int ncolA, long double * pB, int nrowB, int ncolB) {
	int sizeC = nrowA * ncolB;
	long double * pvals , * result;
	result = pvals = (long double *) calloc(sizeC, sizeof(long double));
	if (pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrowA, ncolB);
		return NULL;
		}
	for (int i = 0; i < nrowA; ++ i) {
		for (int j = 0; j < ncolB; ++ j) {
			for (int k = 0; k < ncolA; ++ k) {
				pvals[i * ncolB + j] += pA[i * ncolA + k] * pB[k * ncolB + j];
				}
			}
		}
	if (debuglevel_ >= 2) {
		printf("A x B = C (long double) \n A \n");
		for (int i = 0; i < nrowA; ++ i) {
			for (int j = 0; j < ncolA; ++ j) {
				printf(" %lf ", pA[i * ncolA + j]);
				}
			printf("\n");
			}
		printf("B \n");
		for (int i = 0; i < nrowB; ++ i) {
			for (int j = 0; j < ncolB; ++ j) {
				printf(" %lf ", pB[i * ncolB + j]);
				}
			printf("\n");
			}
		printf("C \n");
		for (int i = 0; i < nrowA; ++ i) {
			for (int j = 0; j < ncolB; ++ j) {
				printf(" %lf ", result[i * ncolB + j]);
				}
			printf("\n");
			}
		}
	return result;
	}

// Funções para cálculo da norma 2 das matrizes em diversas precisões
float fmnorm2(float * pmat, int nrow, int ncol) {
	int size = nrow * ncol;
	float sum = 0;
	while (size -- > 0) {
		float valor = * pmat ++;
		sum += valor * valor;
		}
	return sqrt(sum);
	}

double dmnorm2(double * pmat, int nrow, int ncol) {
	int size = nrow * ncol;
	double sum = 0;
	while (size -- > 0) {
		double valor = * pmat ++;
		sum += valor * valor;
		}
	return sqrt(sum);
	}

long double ldmnorm2(long double * pmat, int nrow, int ncol) {
	int size = nrow * ncol;
	long double sum = 0;
	while (size -- > 0) {
		long double valor = * pmat ++;
		sum += valor * valor;
		}
	return sqrt(sum);
	}

// Funções para leitura das matrizes gravadas pelo MATLAB	
double fgetval(char ** pbuffer) {
// Extrai o primeiro valor existente no "buffer"
	char * pchar = * pbuffer, valor [bufsize], * pvalor = valor;
	while (isblank(* pchar)) {
		pchar ++;
		}
	while (! isblank(* pchar) )	{
		* pvalor ++ = * pchar ++;
		}
	* pvalor = '\0';
	if (debuglevel_ >= 3) {
		printf(" '%s' -> %f", valor, atof(valor));
		}
	* pbuffer = pchar;
	return atof(valor);
	}

double * lermat(const char * fname, int * pnrows, int * pncols) {
// Carrega os dados do arquivo 'fname', gravado pelo MATLAB, numa matriz.
// Retorna a matriz e informa suas dimensões ('nrows' x 'ncols').
// Considera que a matriz está gravada no formato correto.
	// Tenta abrir o arquivo
	char * pchar, * pbuf, buf [bufsize];
	FILE * fp = fopen (fname, "r");
	if (fp == NULL) {
		printf("Não conseguiu ler o arquivo %s! \n", fname);
		return NULL;
		}
	// Despreza as 3 primeiras linhas
	for (int i = 0; i < 3; ++ i) {
		fgets(buf, bufsize, fp);
		}
	// Obtém as dimensões da matriz
	int nrows, ncols;
	fgets(buf, bufsize, fp);
	nrows = atoi(buf + POSROWNBR);
	fgets(buf, bufsize, fp);
	ncols = atoi(buf + POSCOLNBR);
	if (debuglevel_ >= 1) {
		printf("Arquivo %s: linhas = %d, colunas = %d. \n", fname, nrows, ncols);
		}
	double * result, * pval;
	result = pval = (double *) malloc(nrows * ncols * sizeof(double));
	if (pval == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		fclose(fp);
		return NULL;
		}
	for (int i = 0; i < nrows; ++ i) {
		buf[bufsize - 2] = '\0';
		fgets(buf, bufsize, fp);
		if (buf[bufsize - 2] != '\0') {
			printf("A matriz tem colunas demais!");
			fclose(fp);
			return NULL;
			}
		pbuf = buf;
		for (int k = 0; k < ncols; ++ k) {
			double valor;
			valor = fgetval(& pbuf);	
			* pval ++ = valor;
			if (debuglevel_ >= 2) {
				printf(" %f ", valor);
				}
			}
		if (debuglevel_ >= 2) {
			printf("\n");
			}
		}
	* pnrows = nrows;
	* pncols = ncols;
	fclose(fp);
	return result;
	}
