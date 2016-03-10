/*
exercmat.c
Uso:
	exercmat n m [l]
onde
	n é o número do problema
	m é o tamanho do problema
	l (opcional) é o nível de debug a ser usado
Valores de n:
1 <= n <= 3: Problemas normais.
n > 3: Problemas extras. Nesses casos não se estimou o efeito das diversas precisões; apenas a precisão simples foi empregada.
n = 1: Lê duas matrizes geradas pelo MATLAB e calcula o produto e a norma 2 do mesmo em diversas precisões.
n = 2: Lê dois sistemas triangulares gerados pelo MATLAB, resolve-os e calcula a norma 2 do resultado em diversas precisões.
n = 3: Lê um sistema gerado pelo MATLAB, resolve-o pelo método de eliminação de Gauss e calcula o determinante e a norma 2 do resultado em diversas precisões.
n = 4: Lê um sistema gerado pelo MATLAB, resolve-o pelo método de susbtituição LU e calcula o determinante e a norma 2 do resultado.
n = 5: Lê um sistema gerado pelo MATLAB, resolve-o pelo método de susbtituição de Cholesky e calcula o determinante e a norma 2 do resultado.
n = 6: Lê duas matrizes geradas pelo MATLAB e calcula o produto através de rotinas da biblioteca openblas, bem como a norma 2 do mesmo.
n = 7: Lê um sistema gerado pelo MATLAB, resolve-o através da bioblioteca LAPACK e calcula a norma 2 do resultado.

Códigos de retorno:
0: Execução bem-sucedida.
1: Número incorreto de argumentos.
2: Número do problema inválido.
3: Tamanho do problema inválido.
4: Erro na leitura dos dados de entrada.
5: Dados de entrada incompatíveis com uma solução.
6: A matriz não é definida positiva.
7: Erro na alocação de memória.
8: Matriz/sistema singular.
9: Matriz larga demais.

Observações:
1) Compilado e testado com MinGW 4.8.2.
2) Utiliza aritmética com precisão estendida (80 bits). Compilar com a opção -D__USE_MINGW_ANSI_STDIO.
3) Utiliza a biblioteca openblas 2.15.

*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include "cblas.h"
#include "lapacke.h"

#define bufsize 10000			// para leitura dos dados em arquivo
#define FNAME_MAX_SIZE 255
#define POSROWNBR 7				// posição do número de linhas no arquivo
#define POSCOLNBR 10			// posição do número de colunas no arquivo
#define FLOPS_SQRT	15			// https://folding.stanford.edu/home/faq/faq-flops/

void *__gxx_personality_v0;
typedef void f_exec(int);

void calcn2(float * fmat, double * dmat, long double * ldmat, int nrows, int ncols);
void dchangerows(double * pmat, int rows, int ncols, int row1, int row2);
int dfindmax(double * pmat, int nrows, int ncols, int pos, bool colmode, int start);	
double * dmmult(double * pA, int nrowA, int ncolA, double * pB, int nrowB, int ncolB);
double dmnorm2(double * pmat, int nrow, int ncol);
double * dmtrisolve(double * pmat, int nrows, int ncols, bool superior);
void dshowmat(double * pmat, int nrows, int ncols, const char * header);
double * dsolveG(double * psrc, int rank, double * pdet);
double * d2tri(double * psrc, int rank, double * pdet);
void execprob1(int size);
void execprob2(int size);
void execprob3(int size);
void execprob4(int size);
void execprob5(int size);
void execprob6(int size);
void execprob7(int size);
void fchangerows(float * pmat, int rows, int ncols, int row1, int row2);
int ffindmax(float * pmat, int nrows, int ncols, int pos, bool colmode, int start);	
float * fgemm(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB);
double fgetval(char ** pbuffer);
float * fmcopy(double * psrc, int nrows, int ncols);
float * fmmult(float * pA, int nrowA, int ncolA, float * pBf, int nrowB, int ncolB);
float fmnorm2(float * pmat, int nrow, int ncol);
float * fmtrisolve(float * pmat, int nrows, int ncols, bool superior);
int fparseLU(float * pmat, float ** ppL, float ** ppU, float * values,int rank);
void fshowmat(float * pmat, int nrows, int ncols, const char * header);
float * fsolveChol(float * psrc, int rank, float * pdet);
float * fsolveG(float * psrc, int rank, float * pdet);
float * fsolveLS(float * psys, int rank, int nrhs);
float * fsolveLU(float * psrc, int rank, float * pdet);
float * ftranspose(float * psrc, int nrows, int ncols);
float * f2Chol(float * psrc, int rank, float * pdet);
float * f2LU(float * psrc, int rank, float * pdet);
float * f2sys(float * psrc, float * pval, int rank);
float * f2tri(float * psrc, int rank, float * pdet);
void ldchangerows(long double * pmat, int rows, int ncols, int row1, int row2);
int ldfindmax(long double * pmat, int nrows, int ncols, int pos, bool colmode, int start);	
long double * ldmcopy(double * psrc, int nrows, int ncols);
long double * ldmmult(long double * pA, int nrowA, int ncolA, long double * pB, int nrowB, int ncolB);
long double ldmnorm2(long double * pmat, int nrow, int ncol);
long double * ldmtrisolve(long double * pmat, int nrows, int ncols, bool superior);
void ldshowmat(long double * pmat, int nrows, int ncols, const char * header);
long double * ldsolveG(long double * psrc, int rank, long double * pdet);
long double * ld2tri(long double * psrc, int rank, long double * pdet);
double * lermat(const char * fname, int size, int * nrows, int * ncolA);
int main(int argc, const char * argv[]);
void valargs(int argc, const char * argv[], int * pprobnbr, int * psize);


static int debuglevel_ = 0, flops_ = 0;

int main(int argc, const char * argv[]) {
// Executa o problema de acordo com os argumentos passados.
// Retorna 0 se tiver sucesso e um código de erro em caso contrário.
	extern int debuglevel_;
	int probnbr, size;
	// Valida os argumentos passados
	valargs(argc, argv, & probnbr, & size);
	// Executa o problema solicitado
	printf("Solução do problema %d com tamanho %d (nível de debug = %d): \n", probnbr, size, debuglevel_);
	static f_exec * fn[] = {
		& execprob1, & execprob2, & execprob3, 
		& execprob4, & execprob5, & execprob6,
		& execprob7
		};
	fn[probnbr - 1](size);
	return 0;
	}
		
void calcn2(float * fmat, double * dmat, long double * ldmat, int nrows, int ncols) {
// Calcula e relata a norma 2 dos resultados, bem como o esforço computacional necessário
	extern int flops_;
	flops_ = 0;
	float fnorm = fmnorm2(fmat, nrows, ncols);
	double dnorm = dmnorm2(dmat, nrows, ncols);
	long double ldnorm = ldmnorm2(ldmat, nrows, ncols);
	printf("Número de operações necessário para calcular a norma 2: %d. \n", flops_);	
	printf("Norma 2 do resultado: 32 bits = %f, 64 bits = %f, 80 bits = %Lf \n", fnorm, dnorm, ldnorm);
	return;
	}

void valargs(int argc, const char * argv[], int * pprobnbr, int * psize) {
// Valida os argumentos passados ao programa. Informa o número e o tamanho do problema. Define o nível de debug a ser usado.
	extern int debuglevel_;
	if (argc < 3 || argc > 4) {
		printf("Número incorreto de argumentos! \n\t Uso: \n\t\t exercmat probnbr size [level] \n");
		exit(1);
		}
	if (argc == 4)  {
		int level = atoi(argv[3]);
		if (level > 0) {
			debuglevel_ = level;
			}
		else {
			printf("Nível de debug inválido. Valor default assumido. \n");
			}
		}
	int probnbr = atoi(argv[1]);
	int size = atoi(argv[2]);
	if (probnbr < 1 || probnbr > 7) {
		printf("Número do problema inválido (%d)! \n", probnbr);
		exit(2);
		}
	if (size < 0) {
		printf("Tamanho do problema inválido (%d)! \n", size);
		exit(3);
		}
	* pprobnbr = probnbr;
	* psize = size;
	return;
	}

void execprob1(int size) {
// Executa o problema número 2 com o tamanho 'size' indicado.
	// Lê as matrizes de entrada
	extern int flops_;	
	int nrowA, ncolA, nrowB, ncolB;
	double * pAd = lermat("MatA", size, & nrowA, & ncolA);
	double * pBd = lermat("MatB", size, & nrowB, & ncolB );
	// Verifica se podem ser multiplicadas
	if (ncolA != nrowB) {
		printf("As matrizes não podem ser multiplicadas, porque as dimensões são incompatíveis: (%d x %d) e (%d x %d)! \n", nrowA, ncolA, nrowB, ncolB);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	float * pBf = fmcopy(pBd, nrowB, ncolB);
	long double * pAld = ldmcopy(pAd, nrowA, ncolA);
	long double * pBld = ldmcopy(pBd, nrowB, ncolB);
	// Multiplica as matrizes e relata o esforço computacional necessário
	flops_ = 0;
	float * pCf = fmmult(pAf, nrowA, ncolA, pBf, nrowB, ncolB);
	printf("Número de operações para multiplicação das matrizes: %d. \n", flops_);	
	flops_ = 0;
	double * pCd = dmmult(pAd, nrowA, ncolA, pBd, nrowB, ncolB);
	long double * pCld = ldmmult(pAld, nrowA, ncolA, pBld, nrowB, ncolB);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, pCd, pCld, nrowA, ncolB);
	return;
	}
	
void execprob2(int size) {
// Executa o problema número 2 com o tamanho 'size' indicado.
	// Lê as matrizes de entrada
	extern int flops_;	
	int nrowA, ncolA, nrowB, ncolB;
	double * pAd = lermat("TS", size, & nrowA, & ncolA);
	double * pBd = lermat("TI", size, & nrowB, & ncolB );
	// Verifica se possuem as dimensões corretas
	if (ncolA != nrowA + 1) {
		printf("O primeiro sistema não pode ser resolvido, porque as dimensões são incompatíveis: (%d x %d)! \n", nrowA, ncolA);
		exit(5);
		}
	if (ncolB != nrowB + 1) {
		printf("O segundo sistema não pode ser resolvido, porque as dimensões são incompatíveis: (%d x %d)! \n", nrowB, ncolB);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	float * pBf = fmcopy(pBd, nrowB, ncolB);
	long double * pAld = ldmcopy(pAd, nrowA, ncolA);
	long double * pBld = ldmcopy(pBd, nrowB, ncolB);
	// Resolve os sistemas e relata o esforço computacional
	flops_ = 0;
	float * pCf = fmtrisolve(pAf, nrowA, ncolA, true);
	printf("Número de operações para resolver o sistema triangular superior: %d. \n", flops_);	
	flops_ = 0;
	float * pDf = fmtrisolve(pBf, nrowB, ncolB, false);
	printf("Número de operações para resolver o sistema triangular inferior: %d. \n", flops_);	
	flops_ = 0;
	double * pCd = dmtrisolve(pAd, nrowA, ncolA, true);
	double * pDd = dmtrisolve(pBd, nrowB, ncolB, false);
	long double * pCld = ldmtrisolve(pAld, nrowA, ncolA, true);
	long double * pDld = ldmtrisolve(pBld, nrowB, ncolB, false);
	// Calcula e informa a norma 2 dos resultados
	calcn2(pCf, pCd, pCld, 1, nrowA);
	calcn2(pDf, pDd, pDld, 1, nrowB);
	return;
	}
	
void execprob3(int size) {
// Executa o problema número 3 com o tamanho 'size' indicado.
	// Lê o sistema de entrada
	extern int flops_;
	int nrowA, ncolA;
	double * pAd = lermat("S", size, & nrowA, & ncolA);
	// Verifica se pode ser resolvido
	if (ncolA != nrowA + 1) {
		printf("O sistema não podem ser resolvido, porque as dimensões são incompatíveis: (%d x %d)! \n", nrowA, ncolA);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	long double * pAld = ldmcopy(pAd, nrowA, ncolA);
	// Resolve o sistema e relata o valor do determinante e o esforço computacional necessário para solução
	float fdet;
	double ddet;
	long double lddet;
	flops_ = 0;
	float * pCf = fsolveG(pAf, nrowA, & fdet);
	printf("Número de operações necessário para resolver o sistema: %d. \n", flops_);	
	flops_ = 0;
	double * pCd = dsolveG(pAd, nrowA, & ddet);
	long double * pCld = ldsolveG(pAld, nrowA, & lddet);
	printf("Determinante da matriz: 32 bits = %f, 64 bits = %f, 80 bits = %Lf \n", fdet, ddet, lddet);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, pCd, pCld, nrowA, 1);
	return;
	}

void execprob4(int size) {
// Executa o problema número '4' com o tamanho 'size' indicado.
	// Lê o sistema de entrada
	extern int flops_;
	int nrowA, ncolA;
	double * pAd = lermat("S", size, & nrowA, & ncolA);
	// Verifica se pode ser resolvido
	if (ncolA != nrowA + 1) {
		printf("O sistema não podem ser resolvido, porque as dimensões são incompatíveis: (%d x %d)! \n", nrowA, ncolA);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	long double * pAld = ldmcopy(pAd, nrowA, ncolA);
	// Resolve o sistema e relata o valor do determinante e o esforço computacional necessário para solução
	float fdet;
	double ddet;
	long double lddet;
	flops_ = 0;
	float * pCf = fsolveLU(pAf, nrowA, & fdet);
	printf("Número de operações necessário para resolver o sistema: %d. \n", flops_);	
	flops_ = 0;
	printf("Determinante da matriz: 32 bits = %f. \n", fdet);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, NULL, NULL, nrowA, 1);
	return;
	}

void execprob5(int size) {
// Executa o problema número "5" com o tamanho 'size' indicado.
	// Lê o sistema de entrada
	int nrowA, ncolA;
	double * pAd = lermat("S", size, & nrowA, & ncolA);
	// Verifica se pode ser resolvido
	if (ncolA != nrowA + 1) {
		printf("O sistema não podem ser resolvido, porque as dimensões são incompatíveis: (%d x %d)! \n", nrowA, ncolA);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	// Resolve o sistema e relata o valor do determinante e o esforço computacional necessário para solução
	float fdet;
	flops_ = 0;
	float * pCf = fsolveChol(pAf, nrowA, & fdet);
	if (pCf == NULL) {
		exit (6);
		}
	printf("Número de operações necessário para resolver o sistema: %d. \n", flops_);
	flops_ = 0;
	printf("Determinante da matriz: 32 bits = %f. \n", fdet);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, NULL, NULL, nrowA, 1);
	return;
	}

void execprob6(int size) {
// Executa o problema número '6' com o tamanho 'size' indicado.
	// Lê as matrizes de entrada
	int nrowA, ncolA, nrowB, ncolB;
	double * pAd = lermat("MatA", size, & nrowA, & ncolA);
	double * pBd = lermat("MatB", size, & nrowB, & ncolB );
	// Verifica se podem ser multiplicadas
	if (ncolA != nrowB) {
		printf("As matrizes não podem ser multiplicadas, porque as dimensões são incompatíveis: (%d x %d) e (%d x %d)! \n", nrowA, ncolA, nrowB, ncolB);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	// Cria versões em diversas precisões
	float * pBf = fmcopy(pBd, nrowB, ncolB);
	// Multiplica as matrizes
	float * pCf = fgemm(pAf, nrowA, ncolA, pBf, nrowB, ncolB);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, NULL, NULL, nrowA, ncolB);
	return;
	}

void execprob7(int size) {
// Executa o problema número '7' com o tamanho 'size' indicado.
	// Lê o sistema de entrada
	int nrowA, ncolA;
	double * pAd = lermat("S", size, & nrowA, & ncolA);
	// Verifica se pode ser resolvido
	if (ncolA != nrowA + 1) {
		printf("O sistema não podem ser resolvido, porque as dimensões são incompatíveis: (%d x %d)! \n", nrowA, ncolA);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	// Resolve o sistema
	debuglevel_ = 2;
	float * pCf = fsolveLS(pAf, nrowA, 1);
	// Calcula e relata a norma 2 dos resultados
	debuglevel_ = 0;
	calcn2(pCf, NULL, NULL, nrowA, 1);
	return;
	}

	// Wrappers para funções da biblioteca BLAS	
float * fgemm(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB) {
	float * pC = (float *) malloc(nrowA * ncolB * sizeof(float));
	if (pC == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrowA, ncolB);
		exit(7);
		}
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nrowA, ncolB, ncolA, 1, pA, ncolA, pB, ncolB, 0, pC, ncolB);
	return pC;
	}
	
float * fsolveLS(float * psys, int rank, int nrhs) {
	extern int debuglevel_;
	float * pC = (float *) malloc(rank * nrhs * sizeof(float));
	if (pC == NULL) {
		printf("Não conseguiu alocar memória para o resultado %d x %d! \n", rank, nrhs);
		exit(7);
		}
	float * pA = (float *) malloc(rank * rank * sizeof(float));
	if (pA == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank);
		exit(7);
		}
	float * pB = (float *) malloc(rank * nrhs * sizeof(float));
	if (pB == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, nrhs);
		exit(7);
		}
	int ncols = rank + nrhs;
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pA[i * rank + j] = psys[i * ncols + j];
			}
		for (int j = 0; j < nrhs; ++ j) {
			pB[i * nrhs + j] = psys[i * ncols + j + rank];
			}
		}
	if (debuglevel_ == 2) {
		fshowmat(pA, rank, rank, "A");
		fshowmat(pB, rank, nrhs, "B");
		}
	lapack_int retcode = LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', rank, rank + nrhs, nrhs, pA, rank, pB, nrhs);
	if (retcode != 0) {
		printf("Função LAPACKE_sgels retornou erro no argumento %d", -retcode);
		exit(8);
		}
	free(pA);
	free(pB);
	return pC;
	}
	
// Funções para cópia das matrizes em diversas precisões	
float * fmcopy(double * psrc, int nrows, int ncols) {
// Retorna uma cópia em precisão simples (32 bits) da matriz 'psrc'
	float * pdst, * result;
	int size = nrows * ncols;
	result = pdst = (float *) malloc (size * sizeof(float));
	if (pdst == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		exit(7);
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
		exit(7);
		}
	while (size -- > 0) {
		double dvalor;
		long double ldvalor;
		dvalor = * psrc ++;
		ldvalor = dvalor;
		* pdst ++ = ldvalor;
		}
	return result;
	}

	
// Funções para multimplicação das matrizes em diversas precisões
float * fmmult(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB) {
// Retorna o resultado da multiplicação das matrizes A e B em precisão simples.
	extern int debuglevel_;
	int sizeC = nrowA * ncolB;
	float * pvals;
	pvals = (float *) calloc(sizeC, sizeof(float));
	if (pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrowA, ncolB);
		exit(7);
		}
	for (int i = 0; i < nrowA; ++ i) {
		for (int j = 0; j < ncolB; ++ j) {
			for (int k = 0; k < ncolA; ++ k) {
				pvals[i * ncolB + j] += pA[i * ncolA + k] * pB[k * ncolB + j];
				++ flops_;
				}
			}
		}
	if (debuglevel_ >= 2) {
		fshowmat(pA, nrowA, ncolA, "A x B = C (float) \n A");
		fshowmat(pB, nrowB, ncolB, "B");
		fshowmat(pvals, nrowA, ncolB, "C");
		}
	return pvals;
	}
	
double * dmmult(double * pA, int nrowA, int ncolA, double * pB, int nrowB, int ncolB) {
// Retorna o resultado da multiplicação das matrizes A e B em precisão dupla.
	extern int debuglevel_;	
	int sizeC = nrowA * ncolB;
	double * pvals;
	pvals = (double *) calloc(sizeC, sizeof(double));
	if (pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrowA, ncolB);
		exit(7);
		}
	for (int i = 0; i < nrowA; ++ i) {
		for (int j = 0; j < ncolB; ++ j) {
			for (int k = 0; k < ncolA; ++ k) {
				pvals[i * ncolB + j] += pA[i * ncolA + k] * pB[k * ncolB + j];
				++ flops_;				
				}
			}
		}
	if (debuglevel_ >= 2) {
		dshowmat(pA, nrowA, ncolA, "A x B = C (double) \n A");
		dshowmat(pB, nrowB, ncolB, "B");
		dshowmat(pvals, nrowA, ncolB, "C");
		}
	return pvals;
	}

long double * ldmmult(long double * pA, int nrowA, int ncolA, long double * pB, int nrowB, int ncolB) {
// Retorna o resultado da multiplicação das matrizes A e B em precisão estendida.
	extern int debuglevel_;	
	int sizeC = nrowA * ncolB;
	long double * pvals;
	pvals = (long double *) calloc(sizeC, sizeof(long double));
	if (pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrowA, ncolB);
		exit(7);
		}
	for (int i = 0; i < nrowA; ++ i) {
		for (int j = 0; j < ncolB; ++ j) {
			for (int k = 0; k < ncolA; ++ k) {
				pvals[i * ncolB + j] += pA[i * ncolA + k] * pB[k * ncolB + j];
				++ flops_;				
				}
			}
		}
	if (debuglevel_ >= 2) {
		ldshowmat(pA, nrowA, ncolA, "A x B = C (long double) \n A");
		ldshowmat(pB, nrowB, ncolB, "B");
		ldshowmat(pvals, nrowA, ncolB, "C");
		}
	return pvals;
	}


// Funções para solução dos sistemas em diversas precisões
// ... Eliminação Gaussiana com pivotação
float * fsolveG(float * psrc, int rank, float * pdet) {
// Retorna a solução do sistema por Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	float * pTS = f2tri(psrc, rank, pdet);
	float * result = (float *) malloc(rank * sizeof(float));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 1! \n", rank);
		exit(7);
		}
	result = fmtrisolve(pTS, rank, rank + 1, true);
	if (debuglevel_ >= 2) {
		fshowmat(result, rank, 1, "Resultado");
		}
	return result;
	}

float * f2tri(float * psrc, int rank, float * pdet) {
// Retorna o resultado da Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	int ncols = rank + 1;
	float * pval = (float *) calloc(rank * ncols, sizeof(float));
	if (pval == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, ncols);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j <= rank; ++ j) {
			pval[i * ncols + j] = psrc[i * (rank + 1) + j];
			}
		}
	if (debuglevel_ >= 2) {
		fshowmat(pval, rank, ncols, "Inicialização");
		}
	bool sinal = false;
	float det = 1, maxval;
	for (int j = 0; j < rank - 1; ++ j) {
		int maxrow = ffindmax(pval, rank, ncols, j, true, j);
		maxval = pval[maxrow * ncols + j];
		if (maxval == 0) {
			printf("Matriz singular! \n");
			exit(8);
			}
		if (j != maxrow) {
			sinal = ! sinal;
			fchangerows(pval, rank, ncols, j, maxrow);
			if (debuglevel_ >= 2) {
				printf("Coluna %d pivoteamento\n", j);
				fshowmat(pval, rank, ncols, "");
				}
			}
		for (int i = j + 1; i < rank; ++ i) {
			double multiplier = pval[i * ncols + j] / maxval;
			++ flops_;			
			pval[i * ncols + j] = 0;
			for (int k = j + 1; k <= rank; ++ k) {
				pval[i * ncols + k] -= pval[j * ncols + k] * multiplier;
				flops_ += 2;				
				}
			}
		if (debuglevel_ >= 2) {
			printf("Coluna %d eliminação. Pivô = %f\n", j, maxval);
			fshowmat(pval, rank, ncols, "");
			}
		}
	for (int i = 0; i < rank; ++ i) {
		det *= pval[i * ncols + i];
		++ flops_;		
		}
	* pdet = det * (sinal ? -1 : 1);
	return pval;
	}

double * dsolveG(double * psrc, int rank, double * pdet) {
// Retorna a solução do sistema por Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão dupla.
	extern int debuglevel_;
	double * pTS = d2tri(psrc, rank, pdet);
	double * result = (double *) malloc(rank * sizeof(double));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 1! \n", rank);
		exit(7);
		}
	result = dmtrisolve(pTS, rank, rank + 1, true);
	if (debuglevel_ >= 2) {
		dshowmat(result, rank, 1, "Resultado");
		}
	return result;
	}

double * d2tri(double * psrc, int rank, double * pdet) {
// Retorna o resultado da Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão dupla.
	extern int debuglevel_;		
	int ncols = rank + 1;
	double * pval = (double *) calloc(rank * ncols, sizeof(double));
	if (pval == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, ncols);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j <= rank; ++ j) {
			pval[i * ncols + j] = psrc[i * (rank + 1) + j];
			}
		}
	if (debuglevel_ >= 2) {
		dshowmat(pval, rank, ncols, "Inicialização");
		}
	bool sinal = false;
	double det = 1, maxval;
	for (int j = 0; j < rank - 1; ++ j) {
		int maxrow = dfindmax(pval, rank, ncols, j, true, j);
		maxval = pval[maxrow * ncols + j];
		if (maxval == 0) {
			printf("Matriz singular! \n");
			exit(8);
			}
		if (j != maxrow) {
			sinal = ! sinal;
			dchangerows(pval, rank, ncols, j, maxrow);
			if (debuglevel_ >= 2) {
				printf("Coluna %d pivoteamento\n", j);
				dshowmat(pval, rank, ncols, "");
				}
			}
		for (int i = j + 1; i < rank; ++ i) {
			double multiplier = pval[i * ncols + j] / maxval;
			++ flops_;			
			pval[i * ncols + j] = 0;
			for (int k = j + 1; k <= rank; ++ k) {
				pval[i * ncols + k] -= pval[j * ncols + k] * multiplier;
				flops_ += 2;				
				}
			}
		if (debuglevel_ >= 2) {
			printf("Coluna %d eliminação. Pivô = %f\n", j, maxval);
			dshowmat(pval, rank, ncols, "");
			}
		}
	for (int i = 0; i < rank; ++ i) {
		det *= pval[i * ncols + i];
		++ flops_;
		}
	* pdet = det * (sinal ? -1 : 1);
	return pval;
	}

long double * ldsolveG(long double * psrc, int rank, long double * pdet) {
// Retorna a solução do sistema por Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão dupla.
	extern int debuglevel_;
	long double * pTS = ld2tri(psrc, rank, pdet);
	long double * result = (long double *) malloc(rank * sizeof(long double));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 1! \n", rank);
		exit(7);
		}
	result = ldmtrisolve(pTS, rank, rank + 1, true);
	if (debuglevel_ >= 2) {
		ldshowmat(result, rank, 1, "Resultado");
		}
	return result;
	}

long double * ld2tri(long double * psrc, int rank, long double * pdet) {
// Retorna o resultado da Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão estendida.
	extern int debuglevel_;		
	int ncols = rank + 1;
	long double * pval = (long double *) calloc(rank * ncols, sizeof(long double));
	if (pval == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, ncols);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j <= rank; ++ j) {
			pval[i * ncols + j] = psrc[i * (rank + 1) + j];
			}
		}
	if (debuglevel_ >= 2) {
		ldshowmat(pval, rank, ncols, "Inicialização");
		}
	bool sinal = false;
	long double det = 1, maxval;
	for (int j = 0; j < rank - 1; ++ j) {
		int maxrow = ldfindmax(pval, rank, ncols, j, true, j);
		maxval = pval[maxrow * ncols + j];
		if (maxval == 0) {
			printf("Matriz singular! \n");
			exit(8);
			}
		if (j != maxrow) {
			sinal = ! sinal;
			ldchangerows(pval, rank, ncols, j, maxrow);
			if (debuglevel_ >= 2) {
				printf("Coluna %d pivoteamento\n", j);
				ldshowmat(pval, rank, ncols, "");
				}
			}
		for (int i = j + 1; i < rank; ++ i) {
			long double multiplier = pval[i * ncols + j] / maxval;
			++ flops_;
			pval[i * ncols + j] = 0;
			for (int k = j + 1; k <= rank; ++ k) {
				pval[i * ncols + k] -= pval[j * ncols + k] * multiplier;
				flops_ += 2;
				}
			}
		if (debuglevel_ >= 2) {
			printf("Coluna %d eliminação. Pivô = %Lf\n", j, maxval);
			ldshowmat(pval, rank, ncols, "");
			}
		}
	for (int i = 0; i < rank; ++ i) {
		det *= pval[i * ncols + i];
		}
	* pdet = det * (sinal ? -1 : 1);
	return pval;
	}

// ... Substituição LU
int fparseLU(float * pmat, float ** ppL, float ** ppU, float * values, int rank) {
// Extrai as componentes L e U da matriz e carrega os valores 'values', em precisão simples.
	extern int debuglevel_;
	int dstcols = rank + 1, srccols = 2 * rank + 1;
	float * pL = (float *) malloc(rank * dstcols * sizeof(float));
	float * pU = (float *) malloc(rank * dstcols * sizeof(float));
	if (pL == NULL || pU == NULL) {	
		printf("Não conseguiu alocar memória para as matrizes %d x %d! \n", rank, dstcols);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pU[i * dstcols + j] = pmat[i * srccols + j];
			}
		for (int j = 0; j < rank; ++ j) {
			pL[i * dstcols + j] = pmat[i * srccols + j + rank];
			}
		int position = pmat[(i + 1) * srccols - 1];
		pL[i * dstcols + rank] = values[position];
		pU[i * dstcols + rank] = 0;
		}
	* ppL = pL;
	* ppU = pU;
	if (debuglevel_ >= 2) {
		fshowmat(pL, rank, dstcols, "Matriz L");
		fshowmat(pU, rank, dstcols, "Matriz U");
		}
	return 0;
	}
	
float * fsolveLU(float * psrc, int rank, float * pdet) {
// Retorna a solução do sistema por substituição LU e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	float * pmat = f2LU(psrc, rank, pdet);
	float * pval = (float *) malloc(rank * sizeof(float));
	float * result = (float *) malloc(rank * sizeof(float));
	if (result == NULL || pval == NULL) {
		printf("Não conseguiu alocar memória para as matrizes %d x 1! \n", rank);
		exit(7);
		}
	float * pL, * pU;
	for (int i = 0; i < rank; ++ i) {
		pval[i] = psrc[(i + 1) * (rank + 1) - 1]; 
		}
	fparseLU(pmat, & pL, & pU, pval, rank);
	free(pval);
	pval = fmtrisolve(pL, rank, rank + 1, false);
	for (int i = 0; i < rank; ++ i) {
		pU[(i + 1) * (rank + 1) - 1] = pval[i]; 
		}
	result = fmtrisolve(pU, rank, rank + 1, true);
	if (debuglevel_ >= 2) {
		fshowmat(result, rank, 1, "Resultado");
		}
	return result;
	}

float * f2LU(float * psrc, int rank, float * pdet) {
// Retorna o resultado da substituição LU e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;		
	int ncols = 2 * rank + 1;
	float * pval = (float *) calloc(rank * ncols, sizeof(float));
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pval[i * ncols + j] = psrc[i * (rank + 1) + j];
			}
		pval[(i + 1) * ncols - 1] = i;
		}
	if (debuglevel_ >= 2) {
		fshowmat(pval, rank, ncols, "Inicialização");
		}
	bool sinal = false;
	float det = 1, maxval;
	for (int j = 0; j < rank - 1; ++ j) {
		int maxrow = ffindmax(pval, rank, ncols, j, true, j);
		maxval = pval[maxrow * ncols + j];
		if (maxval == 0) {
			printf("Matriz singular! \n");
			exit(8);
			}
		if (j != maxrow) {
			sinal = ! sinal;
			fchangerows(pval, rank, ncols, j, maxrow);
			if (debuglevel_ >= 2) {
				printf("Coluna %d pivoteamento\n", j);
				fshowmat(pval, rank, ncols, "");
				}
			}
		for (int i = j + 1; i < rank; ++ i) {
			double multiplier = pval[i * ncols + j] / maxval;
			++ flops_;			
			pval[i * ncols + j] = 0;
			pval[i * ncols + rank + j] = multiplier;
			for (int k = j + 1; k < rank; ++ k) {
				pval[i * ncols + k] -= pval[j * ncols + k] * multiplier;
				flops_ += 2;				
				}
			}
		if (debuglevel_ >= 2) {
			printf("Coluna %d eliminação. Pivô = %f\n", j, maxval);
			fshowmat(pval, rank, ncols, "");
			}
		}
	for (int i = 0; i < rank; ++ i) {
		det *= pval[i * ncols + i];
		++ flops_;		
		pval[i * ncols + rank + i] = 1;
		}
	* pdet = det * (sinal ? -1 : 1);
	return pval;
	}

// ... Substituição de Cholesky
float * fsolveChol(float * psrc, int rank, float * pdet) {
// Retorna a solução do sistema por substituição de Cholesky e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	debuglevel_ = 2;
	float * pL = f2Chol(psrc, rank, pdet);
	if (pL == NULL) {
		return NULL;
		}
	float * pval = (float *) malloc(rank * sizeof(float));
	float * result = (float *) malloc(rank * sizeof(float));
	if (result == NULL || pval == NULL) {
		printf("Não conseguiu alocar memória para as matrizes %d x 1! \n", rank);
		exit(7);
		}
	float * pU = ftranspose(pL, rank, rank);
	if (debuglevel_ >= 2) {
		fshowmat(pU, rank, rank, "Transposta");
		}
	for (int i = 0; i < rank; ++ i) {
		pval[i] = psrc[(i + 1) * (rank + 1) - 1]; 
		}
	float * pfL = f2sys(pL, pval, rank);
	free(pval);
	pval = fmtrisolve(pfL, rank, rank + 1, false);
	float * pfU = f2sys(pU, pval, rank);
	result = fmtrisolve(pfU, rank, rank + 1, true);
	if (debuglevel_ >= 2) {
		fshowmat(result, rank, 1, "Resultado");
		}
	debuglevel_ = 0;
	return result;
	}

float * f2Chol(float * psrc, int rank, float * pdet) {
// Retorna o resultado da substituição de Cholesky e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;		
	float * pval = (float *) calloc(rank * rank, sizeof(float));
	if (pval == NULL) {
		printf("Não conseguiu alocar memória para a matriz estendida %d x %d! \n", rank, rank);
		exit(7);
		}
	float det = 1;
	for (int i = 0; i < rank; ++ i) {
		float sum;
		for (int j = 0; j <= i; ++ j) {
			sum = 0;
            for (int k = 0; k < j; ++ k) {
				sum += pval[i * rank + k] * pval[j * rank + k];
				flops_ += 2;
				}
            if (i == j) {
				float quadrado = psrc[i * (rank + 2)] - sum;
				if (quadrado <= 0) {
					printf("A matriz não é definida positiva!");
					return NULL;
					}
				float valor = sqrt(quadrado);
				pval[i * (rank + 1)] = valor;
				det *= valor;
				flops_ += 2 + FLOPS_SQRT;
				if (debuglevel_ >= 3) {
					printf("L%d%d = %f \t", i, j, valor);
					}				
				}
			else {
				float valor = (psrc[i * (rank + 1) + j] - sum) / pval[j * (rank + 1)];
				pval[i * rank + j] = valor;
				flops_ += 2;
				if (debuglevel_ >= 3) {
					printf("L%d%d = %f \t", i, j, valor);
					}				
				}
			}
		}
	if (debuglevel_ >= 3) {
		printf("\n");
		}				
	if (debuglevel_ >= 2) {
		fshowmat(pval, rank, rank, "Decomposição de Cholesky");
		}
	* pdet = det * det;
	++ flops_;
	return pval;
	}
	
// ... auxiliares para vários métodos
int ffindmax(float * pmat, int nrows, int ncols, int pos, bool colmode, int start) {
// Retorna o número da linha que possui o maior valor absoluto na posição indicada, em precisão simples.
	float maxval = 0;
	int i, address, maxpos = start;
	int size = colmode ? nrows : ncols;
	for (i = start; i < size; ++ i) {
		if (colmode) {
			address = i * ncols + pos;
			}
		else {
			address = pos * nrows + i;
			}
		float value = fabs(pmat[address]);
		if (maxval < value) {
			maxval = value;
			maxpos = i;
			}
		}
	return maxpos;
	}
	
void fchangerows(float * pmat, int rows, int ncols, int row1, int row2) {
// Troca duas linhas de posição, em precisão simples.
	int row1pos = row1 * ncols;
	int row2pos = row2 * ncols;
	for (int j = 0; j < ncols; ++ j) {
		float value = pmat[row1pos + j];
		pmat[row1pos + j] = pmat[row2pos + j];
		pmat[row2pos + j] = value;
		}
	}

int dfindmax(double * pmat, int nrows, int ncols, int pos, bool colmode, int start) {
// Retorna o número da linha que possui o maior valor absoluto na posição indicada, em precisão dupla.
	double maxval = 0;
	int i, address, maxpos = start;
	int size = colmode ? nrows : ncols;
	for (i = start; i < size; ++ i) {
		if (colmode) {
			address = i * ncols + pos;
			}
		else {
			address = pos * nrows + i;
			}
		double value = fabs(pmat[address]);
		if (maxval < value) {
			maxval = value;
			maxpos = i;
			}
		}
	return maxpos;
	}
	
void dchangerows(double * pmat, int rows, int ncols, int row1, int row2) {
// Troca duas linhas de posição, em precisão dupla.
	int row1pos = row1 * ncols;
	int row2pos = row2 * ncols;
	for (int j = 0; j < ncols; ++ j) {
		double value = pmat[row1pos + j];
		pmat[row1pos + j] = pmat[row2pos + j];
		pmat[row2pos + j] = value;
		}
	}

int ldfindmax(long double * pmat, int nrows, int ncols, int pos, bool colmode, int start) {
// Retorna o número da linha que possui o maior valor absoluto na posição indicada, em precisão estendida.	
	long double maxval = 0;
	int i, address, maxpos = start;
	int size = colmode ? nrows : ncols;
	for (i = start; i < size; ++ i) {
		if (colmode) {
			address = i * ncols + pos;
			}
		else {
			address = pos * nrows + i;
			}
		long double value = fabs(pmat[address]);
		if (maxval < value) {
			maxval = value;
			maxpos = i;
			}
		}
	return maxpos;
	}
	
void ldchangerows(long double * pmat, int rows, int ncols, int row1, int row2) {
// Troca duas linhas de posição, em precisão estendida.
	int row1pos = row1 * ncols;
	int row2pos = row2 * ncols;
	for (int j = 0; j < ncols; ++ j) {
		long double value = pmat[row1pos + j];
		pmat[row1pos + j] = pmat[row2pos + j];
		pmat[row2pos + j] = value;
		}
	}

float * ftranspose(float * psrc, int nrows, int ncols) {
// Retorna a transposta da matriz, em precisão simples.
	float * pdst = (float *) malloc(nrows * ncols * sizeof(float));
	if (pdst == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < ncols; ++ j) {
			pdst[i * nrows + j] = psrc[j * ncols + i];
			}
 		}
	return pdst;
	}

float * f2sys(float * psrc, float * pval, int rank) {
// Monta um sistema a partir de uma matriz quadrada e um vetor de valores, em precisão simples.
	extern int debuglevel_;
	float * pdst = (float *) malloc(rank * (rank + 1) * sizeof(float));
	if (pdst == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank + 1);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pdst[i * (rank + 1) + j] = psrc[i * rank + j]; 
			}
		pdst[(i + 1) * (rank + 1) - 1] = pval[i]; 
		}
	if (debuglevel_ >= 2) {
		fshowmat(pdst, rank, rank + 1, "Sistema");
		}
	return pdst;
	}
	

// Funções para resover sistemas triangulares em diversas precisões
float * fmtrisolve(float * pmat, int nrows, int ncols, bool superior) {
	float * result;
	result = (float *) malloc(nrows * sizeof(float));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		exit(7);
		}
	if (superior) {
		for (int i = nrows - 1; i >= 0; -- i) {
			float divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				exit(8);
				}
			float sum = 0;
			for (int j = i + 1; j < nrows; ++ j) {
				float coef = pmat[i * ncols + j];
				float x = result[j];
				sum += coef * x;
				flops_ += 2;				
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			float parm = pmat[(i + 1) * ncols - 1];
			float valor = (parm - sum) / divisor;
			++ flops_;			
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
				exit(8);
				}
			float sum = 0;
			for (int j = 0; j < i; ++ j) {
				float coef = pmat[i * ncols + j];
				float x = result[j];
				sum += coef * x;
				flops_ += 2;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			float parm = pmat[(i + 1) * ncols - 1];
			float valor = (parm - sum) / divisor;
			++ flops_;
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
		exit(7);
		}
	if (superior) {
		for (int i = nrows - 1; i >= 0; -- i) {
			double divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				exit(8);
				}
			double sum = 0;
			for (int j = i + 1; j < nrows; ++ j) {
				double coef = pmat[i * ncols + j];
				double x = result[j];
				sum += coef * x;
				flops_ += 2;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			double parm = pmat[(i + 1) * ncols - 1];
			double valor = (parm - sum) / divisor;
			++ flops_;			
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
				exit(8);
				}
			double sum = 0;
			for (int j = 0; j < i; ++ j) {
				double coef = pmat[i * ncols + j];
				double x = result[j];
				sum += coef * x;
				flops_ += 2;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %f = %f * %f \n", j, sum, coef, x);
					}
				}
			double parm = pmat[(i + 1) * ncols - 1];
			double valor = (parm - sum) / divisor;
			++ flops_;			
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
		exit(7);
		}
	if (superior) {
		for (int i = nrows - 1; i >= 0; -- i) {
			long double divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				exit(8);
				}
			long double sum = 0;
			for (int j = i + 1; j < nrows; ++ j) {
				long double coef = pmat[i * ncols + j];
				long double x = result[j];
				sum += coef * x;
				flops_ += 2;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %Lf = %Lf * %Lf \n", j, sum, coef, x);
					}
				}
			long double parm = pmat[(i + 1) * ncols - 1];
			long double valor = (parm - sum) / divisor;
			++ flops_;
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %Lf = (%Lf - %Lf) / %Lf \n", i, valor, parm, sum, divisor);
				}
			}
		}
	else {
		for (int i = 0; i < nrows; ++ i) {
			long double divisor = pmat[i * (ncols + 1)];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				exit(8);
				}
			long double sum = 0;
			for (int j = 0; j < i; ++ j) {
				long double coef = pmat[i * ncols + j];
				long double x = result[j];
				flops_ += 2;
				sum += coef * x;
				if (debuglevel_ >= 3) {
					printf("Coluna %d: %Lf = %Lf * %Lf \n", j, sum, coef, x);
					}
				}
			long double parm = pmat[(i + 1) * ncols - 1];
			long double valor = (parm - sum) / divisor;
			++ flops_;
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %Lf = (%Lf - %Lf) / %Lf \n", i, valor, parm, sum, divisor);
				}
			}
		}
	return result;
	}

	
// Funções para cálculo da norma 2 das matrizes em diversas precisões
float fmnorm2(float * pmat, int nrow, int ncol) {
	if (pmat == NULL) {
		return 0;
		}
	int size = nrow * ncol;
	float sum = 0;
	while (size -- > 0) {
		float valor = * pmat ++;
		sum += valor * valor;
		flops_ += 2;
		}
	return sqrt(sum);
	}

double dmnorm2(double * pmat, int nrow, int ncol) {
	if (pmat == NULL) {
		return 0;
		}
	int size = nrow * ncol;
	double sum = 0;
	while (size -- > 0) {
		double valor = * pmat ++;
		sum += valor * valor;
		flops_ += 2;		
		}
	return sqrt(sum);
	}

long double ldmnorm2(long double * pmat, int nrow, int ncol) {
	if (pmat == NULL) {
		return 0;
		}
	int size = nrow * ncol;
	long double sum = 0;
	while (size -- > 0) {
		long double valor = * pmat ++;
		sum += valor * valor;
		flops_ += 2;		
		}
	return sqrt(sum);
	}

// Funções para exibição de matrizes em diversas precisões
void fshowmat(float * pmat, int nrows, int ncols, const char * header) {
	if (header != NULL && strlen(header) > 0) {
		printf("%s \n", header);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < ncols; ++ j) {
			printf(" %f ", pmat[i * ncols + j]);
			}
		printf("\n");
		}
	}

void dshowmat(double * pmat, int nrows, int ncols, const char * header) {
	if (header != NULL && strlen(header) > 0) {
		printf("%s \n", header);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < ncols; ++ j) {
			printf(" %f ", pmat[i * ncols + j]);
			}
		printf("\n");
		}
	}

void ldshowmat(long double * pmat, int nrows, int ncols, const char * header) {
	if (header != NULL && strlen(header) > 0) {
		printf("%s \n", header);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < ncols; ++ j) {
			printf(" %f ", pmat[i * ncols + j]);
			}
		printf("\n");
		}
	}
	
// Funções para leitura das matrizes gravadas pelo MATLAB	
double fgetval(char ** pbuffer) {
// Extrai o primeiro valor existente no "buffer"
	extern int debuglevel_;
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

double * lermat(const char * fname, int size, int * pnrows, int * pncols) {
// Carrega os dados do arquivo 'fname''size', gravado pelo MATLAB, numa matriz.
// Retorna a matriz e informa suas dimensões ('nrows' x 'ncols').
// Considera que a matriz está gravada no formato correto.
	// Tenta abrir o arquivo
	extern int debuglevel_;
	char name[FNAME_MAX_SIZE + 1];
	sprintf(name, "%s%d", fname, size);
	FILE * fp = fopen (name, "r");
	if (fp == NULL) {
		printf("Não conseguiu ler o arquivo %s! \n", name);
		exit(4);
		}
	char * pbuf, buf[bufsize];
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
		printf("Arquivo %s: linhas = %d, colunas = %d. \n", name, nrows, ncols);
		}
	double * result, * pval;
	result = pval = (double *) malloc(nrows * ncols * sizeof(double));
	if (pval == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		fclose(fp);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		buf[bufsize - 2] = '\0';
		fgets(buf, bufsize, fp);
		if (buf[bufsize - 2] != '\0') {
			printf("A matriz tem colunas demais!");
			fclose(fp);
			exit(9);
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
