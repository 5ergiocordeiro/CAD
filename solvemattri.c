/*
solvemattri.c
Lê dois sistemas triangulares gerados pelo MATLAB, resolve-os e calcula a norma 2 do resultado em diversas precisões.
Uso:
	solvemattri
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
double dmnorm2(double * pmat, int nrow, int ncol);
double * dmtrisolve(double * pmat, int nrows, int ncols, bool superior);
float fmnorm2(float * pmat, int nrow, int ncol);
long double ldmnorm2(long double * pmat, int nrow, int ncol);
float * fmcopy(double * psrc, int nrows, int ncols);
float * fmtrisolve(float * pmat, int nrows, int ncols, bool superior);
long double * ldmcopy(double * psrc, int nrows, int ncols);
long double * ldmtrisolve(long double * pmat, int nrows, int ncols, bool superior);
double * lermat(const char * fname, int * nrows, int * ncolA);
int main(int argc, const char * argv[]);

static int debuglevel_ = 0;

int main(int argc, const char * argv[]) {
	// Lê as matrizes de entrada
	float * pAf, * pBf, * pCf, * pDf, fnormC, fnormD;
	double * pAd, * pBd, * pCd, * pDd, dnormC, dnormD ;
	long double * pAld, * pBld, * pCld, * pDld, ldnormC, ldnormD;
	// __float128 * pAq, * pBq, * pCq, qnorm;
	int nrowA, ncolA, nrowB, ncolB, nrowC, ncolC;
	pAd = lermat("TS", & nrowA, & ncolA);
	if (pAd == NULL) {
		return 1;
		}
	pBd = lermat("TI", & nrowB, & ncolB );
	if (pBd == NULL) {
		return 1;
		}
	// Verifica se possuem as dimensões corretas
	if (ncolA != nrowA + 1) {
		printf("O primeiro sistema não pode ser resolvido, porque as dimensões são incompatíveis: (%d,%d)", nrowA, ncolA);
		return 2;
		}
	if (ncolB != nrowB + 1) {
		printf("O segundo sistema não pode ser resolvido, porque as dimensões são incompatíveis: (%d,%d)", nrowB, ncolB);
		return 2;
		}
	// Cria versões em diversas precisões
	pAf = fmcopy(pAd, nrowA, ncolA);
	pBf = fmcopy(pBd, nrowB, ncolB);
	pAld = ldmcopy(pAd, nrowA, ncolA);
	pBld = ldmcopy(pBd, nrowB, ncolB);
	// Resolve os sistemas
	pCf = fmtrisolve(pAf, nrowA, ncolA, true);
	pDf = fmtrisolve(pBf, nrowB, ncolB, false);
	pCd = dmtrisolve(pAd, nrowA, ncolA, true);
	pDd = dmtrisolve(pBd, nrowB, ncolB, false);
	pCld = ldmtrisolve(pAld, nrowA, ncolA, true);
	pDld = ldmtrisolve(pBld, nrowB, ncolB, false);
	// Calcula a norma 2 dos resultados
	fnormC = fmnorm2(pCf, 1, nrowA);
	fnormD = fmnorm2(pDf, 1, nrowB);
	dnormC = dmnorm2(pCd, 1, nrowA);
	dnormD = dmnorm2(pDd, 1, nrowB);
	ldnormC = ldmnorm2(pCld, 1, nrowA);
	ldnormD = ldmnorm2(pDld, 1, nrowB);
	printf("Norma 2 do resultado: 32 bits = %f e %f, 64 bits = %f e %f, 80 bits = %f e %f", fnormC, fnormD, dnormC, dnormD, (double) ldnormC, (double) ldnormD);	
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

// Funções para resover sistemas triangulares em diversas precisões
float * fmtrisolve(float * pmat, int nrows, int ncols, bool superior) {
	float * result;
	result = (float *) malloc(nrows * sizeof(float));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		return NULL;
		}
	if (superior) {
		for (int i = nrows - 1; i >= 0; -- i) {
			float divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				return NULL;
				}
			float sum = 0;
			for (int j = i + 1; j < nrows; ++ j) {
				float coef = pmat[i * ncols + j];
				float x = result[j];
				sum += coef * x;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			float parm = pmat[(i + 1) * ncols - 1];
			float valor = (parm - sum) / divisor;
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %f = (%f - %f) / %f \n", i, valor, parm, sum, divisor);
				}
			}
		}
	else {
		for (int i = 0; i < nrows; ++ i) {
			float divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				return NULL;
				}
			float sum = 0;
			for (int j = 0; j < i; ++ j) {
				float coef = pmat[i * ncols + j];
				float x = result[j];
				sum += coef * x;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			float parm = pmat[(i + 1) * ncols - 1];
			float valor = (parm - sum) / divisor;
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %f = (%f - %f) / %f \n", i, valor, parm, sum, divisor);
				}
			}
		}
	return result;
	}

double * dmtrisolve(double * pmat, int nrows, int ncols, bool superior) {
	double * result;
	result = (double *) malloc(nrows * sizeof(double));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		return NULL;
		}
	if (superior) {
		for (int i = nrows - 1; i >= 0; -- i) {
			double divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				return NULL;
				}
			double sum = 0;
			for (int j = i + 1; j < nrows; ++ j) {
				double coef = pmat[i * ncols + j];
				double x = result[j];
				sum += coef * x;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			double parm = pmat[(i + 1) * ncols - 1];
			double valor = (parm - sum) / divisor;
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %f = (%f - %f) / %f \n", i, valor, parm, sum, divisor);
				}
			}
		}
	else {
		for (int i = 0; i < nrows; ++ i) {
			double divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				return NULL;
				}
			double sum = 0;
			for (int j = 0; j < i; ++ j) {
				double coef = pmat[i * ncols + j];
				double x = result[j];
				sum += coef * x;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			double parm = pmat[(i + 1) * ncols - 1];
			double valor = (parm - sum) / divisor;
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %f = (%f - %f) / %f \n", i, valor, parm, sum, divisor);
				}
			}
		}
	return result;
	}

long double * ldmtrisolve(long double * pmat, int nrows, int ncols, bool superior) {
	long double * result;
	result = (long double *) malloc(nrows * sizeof(long double));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		return NULL;
		}
	if (superior) {
		for (int i = nrows - 1; i >= 0; -- i) {
			long double divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				return NULL;
				}
			long double sum = 0;
			for (int j = i + 1; j < nrows; ++ j) {
				long double coef = pmat[i * ncols + j];
				long double x = result[j];
				sum += coef * x;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			long double parm = pmat[(i + 1) * ncols - 1];
			long double valor = (parm - sum) / divisor;
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %f = (%f - %f) / %f \n", i, valor, parm, sum, divisor);
				}
			}
		}
	else {
		for (int i = 0; i < nrows; ++ i) {
			long double divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				return NULL;
				}
			long double sum = 0;
			for (int j = 0; j < i; ++ j) {
				long double coef = pmat[i * ncols + j];
				long double x = result[j];
				sum += coef * x;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			long double parm = pmat[(i + 1) * ncols - 1];
			long double valor = (parm - sum) / divisor;
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %f = (%f - %f) / %f \n", i, valor, parm, sum, divisor);
				}
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
