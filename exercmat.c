/*
/*
/*
exercmat.c
Uso:
	exercmat n m [p] [l] [i] [e]
onde
	n é o número do problema
	m é o tamanho do problema
	p (opcional) indica se deve ou não ser usado precondicionador
	l (opcional) é o nível de debug a ser usado
	i (opcional) é o número máximo de iterações admitidas
	e (opcional) é o módulo do erro admitido

Valores de n:
1 <= n <= 3: Problemas normais.
n > 3: Problemas extras. Nesses casos não se estimou o efeito das diversas precisões; apenas a precisão simples foi empregada.
n = 1: Lê duas matrizes geradas pelo MATLAB e calcula o produto e a norma 2 do mesmo em diversas precisões.
n = 2: Lê dois sistemas triangulares gerados pelo MATLAB, resolve-os e calcula a norma 2 do resultado em diversas precisões.
n = 3: Lê um sistema gerado pelo MATLAB, resolve-o pelo método de eliminação de Gauss e calcula o determinante e a norma 2 do resultado em diversas precisões.
n = 4: Lê um sistema gerado pelo MATLAB, resolve-o pelo método de susbtituição LU e calcula o determinante e a norma 2 do resultado.
n = 5: Lê um sistema gerado pelo MATLAB, resolve-o pelo método de susbtituição de Cholesky e calcula o determinante e a norma 2 do resultado.
n = 6: Lê duas matrizes geradas pelo MATLAB e calcula o produto através de diversas rotinas, comparando o desempenho.
n = 7: Lê um sistema gerado pelo MATLAB, resolve-o através da bioblioteca LAPACK e calcula a norma 2 do resultado.
n = 8: Lê um sistema gerado pelo MATLAB, resolve-o pelo método de diagonalização e calcula o determinante e a norma 2 do resultado.
n = 9: Lê um sistema gerado pelo MATLAB, resolve-o pelo método do cálculo da matriz inversa por eliminação de Gauss e calcula o determinante e a norma 2 do resultado.
n = 10: Lê um sistema gerado pelo MATLAB e retorna a equação característica da matriz, o determinante e a inversa.
n = 11: Lê um sistema gerado pelo MATLAB, resolve-o pelo método de decomposição LU, verifica a precisão conforme a tolerância permitida e aplica a correção, se necessário.
n = 12: Lê um sistema gerado pelo MATLAB e o resolve pelo método iterativo de Jacobi.
n = 13: Lê um sistema gerado pelo MATLAB e o resolve pelo método de Gauss-Seidel.
n = 14: Lê uma matriz gerada pelo MATLAB e calcula o maior e o menor autovalor pelo método das potências.
n = 15: Lê uma matriz gerada pelo MATLAB e calcula todos os seus autovalores pelo método de Jacobi.

Valores de p:
p = 0: Não usar precondicionador(default)
p = 1: Usar precondicionador Jacobiano

Valores de l: Inteiro não negativo

Valores de i: Inteiro positivo 

Valores de e: Real positivo


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
10: Erro retornado pela biblioteca LAPACK/LAPACKE.
11: Método iterativo demorou demais para convergir.
12: Método iterativo divergiu.
13: A matriz não é simétrica.

Observações:
1) Compilado e testado com MinGW 4.8.2.
2) Utiliza a biblioteca OpenBlas 2.15.
3) Utiliza a biblioteca LAPACK 3.6.0. Compilar com as opções -D__USE_MINGW_ANSI_STDIO e -DHAVE_LAPACK_CONFIG_H -DLAPACK_COMPLEX_CPP e linkar com compilador Fortran (gfortran).


TO DO:
1) Verificar liberação de memória alocada
2) Testar função fpower.
4) Verificar aumento do esforço com aumento do tamanho:
	Gauss x Cholesky
	Leverrier x Leverrier-Faddeev
*/

#define __USE_MINGW_ANSI_STDIO 1	// para usar precisão estendida
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <omp.h>
#include <sys/time.h>
#include <windows.h>
#include <time.h>
#include "cblas.h"

#define HAVE_LAPACK_CONFIG_H 1
#define LAPACK_COMPLEX_CPP 1
#include "lapacke.h"

#define bufsize 10000				// para leitura dos dados em arquivo
#define FNAME_MAX_SIZE 255
#define POSROWNBR 7					// posição do número de linhas no arquivo
#define POSCOLNBR 10				// posição do número de colunas no arquivo
#define FLOPS_SQRT	15				// https://folding.stanford.edu/home/faq/faq-flops/
#define FLOPS_DIV	4				// https://stackoverflow.com/questions/329174/what-is-flop-s-and-is-it-a-good-measure-of-performance
#define FLOPS_MMULT_EXP	3
#define DEBUGLEVEL_DEF	0			// nível de debug default
#define MAXERR_DEF	1e-5			// valor de erro máximo default
#define MAXITER_DEF	10				// número de iterações máximo default

void *__gxx_personality_v0;			// desabilita tratamento de exceção

typedef void f_exec(int);			// função a ser despachada
typedef int f_iter(float *, float *, float *, float *, float **, float *, float *, int *, int);

// Protótipos de funções
void calcn2(float * fmat, double * dmat, long double * ldmat, int nrows, int ncols);
void dchangerows(double * pmat, int rows, int ncols, int row1, int row2);
int dfindmax(double * pmat, int nrows, int ncols, int pos, bool colmode, int start);	
double * dmmult(double * pA, int nrowA, int ncolA, double * pB, int nrowB, int ncolB);
double dmnorm2(double * pmat, int nrow, int ncol);
double * dmtrisolve(double * pmat, int nrows, int ncols, bool superior);
void dshowmat(double * pmat, int nrows, int ncols, const char * header);
double * dsolveG(double * psrc, int rank, double * pdet);
double * d2tri(double * psrc, int rank, double * pdet);
f_exec execprob1, execprob2, execprob3, execprob4, execprob5,
	execprob6, execprob7, execprob8, execprob9, execprob10,
	execprob11, execprob12, execprob13, execprob14, execprob15;
void fchangerows(float * pmat, int rows, int ncols, int row1, int row2);
float * fdoLU(float * pB, float * pL, float * pU, int * pP, int nrows);
float * feqcaracL(float * pmat, int nrows, int ncols);
float * feqcaracLF(float * pmat, int nrows, int ncols, float * pdev = NULL, float ** ppinv = NULL);
int ffindmax(float * pmat, int nrows, int ncols, int pos, bool colmode, int start);	
void ffromsys(float * psys, int nrows, int ncols, float ** ppA, float ** ppB);
float * fgemm(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB);
float * fgemmref(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB);
double fgetval(char ** pbuffer);
float * fident(int rank, float val = 1);
float * finvG(float * pmat, int rank, int ncols, float * pdet);
float * finvTS(float * pmat, int rank, int ncols, bool superior);
bool fisddom(float * pmat, int nrows, int ncols);
bool fissym(float * pmat, int nrows, int ncols);
f_iter fiterGS, fiterJ, fiterLU;
int fiterate(int type, f_iter * pfn, float * pA, float * pB, float ** ppX, int * piter, float * perror, float * pL, float * pU, int * pP, int nrows);
float * fmadd(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB, bool add = true);
int fmavJ(float * pmat, int nrows, int ncols, float ** ppav, int * piter);
int fmavP(float * pmat, int nrows, int ncols, float * pmax, float * pmin, int * piter);
int fmmaxavP(float * pmat, int nrows, int ncols, float * pmax, int * piter, bool direto = true);
float * fmcopy(double * psrc, int nrows, int ncols);
float * fmmult(float * pA, int nrowA, int ncolA, float * pBf, int nrowB, int ncolB);
float fmnormi(float * fmerror, int nrows, int ncols);
float fmnorm2(float * pmat, int nrow, int ncol);
float * fmtimes(float * pmat, int nrows, int ncols, float value);
float * fmtrisolve(float * pmat, int nrows, int ncols, bool superior);
int fmakeLU(float * pmat, int nrows, int ncols, float * values, int * position);
float * fpower (float * pmat, int nrows, int ncols);
void fshowmat(float * pmat, int nrows, int ncols, const char * header);
float * fsolveChol(float * psrc, int rank, float * pdet = NULL);
float * fsolveDG(float * psrc, int rank, float * pdet);
float * fsolveG(float * psrc, int rank, float * pdet = NULL);
int fsolveGS(float * psrc, int rank, float ** ppX, int * piter);
int fsolveJ(float * psrc, int rank, float ** ppX, int * piter);
float * fsolveLS(float * psys, int rank, int nrhs);
int fsolveLU(float * psys, int nrows, int ncols, float ** ppX, float * pdet = NULL, int * pinter = NULL, float * perror = NULL);
float ftrace(float * pmat, int nrows, int ncols);
float * ftranspose(float * psrc, int nrows, int ncols);
float * f2Chol(float * psrc, int rank, float * pdet);
float * f2diag(float * psrc, int rank, float * pdet);
void f2LU(float * psrc, int rank, float ** ppL, float ** ppU, int ** ppP, float * pdet);
float * f2sys(float * psrc, float * pval, int rank);
float * f2tri(float * psrc, int rank, int ncols, float * pdet = NULL);
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
void ucrono(bool init, int divisor);
void valargs(int argc, const char * argv[], int * pprobnbr, int * psize);


static int debuglevel_, flops_, maxiter_, prec_;
static float maxerr_;

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
		& execprob7, & execprob8, & execprob9,
		& execprob10, & execprob11, & execprob12,
		& execprob13, & execprob14, & execprob15,
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
	extern float maxerr_;
	if (argc < 3 || argc > 7) {
		printf("Número incorreto de argumentos! \n\t Uso: \n\t\t exercmat probnbr size [level] \n");
		exit(1);
		}
	prec_ = 0;
	if (argc >= 4) {
		float prec = atoi(argv[3]);
		if (prec >= 0 && prec <= 1) {
			prec_ = prec;
			}
		else {
			printf("Precondicionador inválido. Valor default assumido. \n");
			}
		}
	debuglevel_ = DEBUGLEVEL_DEF;
	if (argc >= 5) {
		int level = atoi(argv[4]);
		if (level > 0) {
			debuglevel_ = level;
			}
		else {
			printf("Nível de debug inválido. Valor default assumido. \n");
			}
		}
	maxiter_ = MAXITER_DEF;
	if (argc >= 6) {
		int niter = atoi(argv[5]);
		if (niter > 0) {
			maxiter_ = niter;
			}
		else {
			printf("Número de iterações máximo inválido. Valor default assumido. \n");
			}
		}
	maxerr_ = MAXERR_DEF;
	if (argc >= 7) {
		float error = atof(argv[6]);
		if (error > 0) {
			maxerr_ = error;
			}
		else {
			printf("Valor de erro máximo inválido. Valor default assumido. \n");
			}
		}
	int probnbr = atoi(argv[1]);
	int size = atoi(argv[2]);
	if (probnbr < 1 || probnbr > 15) {
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


// Funções despachadas
void execprob1(int size) {
// Executa o problema número 1 com o tamanho 'size' indicado.
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
	float * pCf = fsolveG(pAf, nrowA);
	printf("Número de operações necessário para resolver o sistema: %d. \n", flops_);	
	flops_ = 0;
	free(pCf);
	pCf = fsolveG(pAf, nrowA, & fdet);
	printf("Número de operações necessário para resolver o sistema com determinante: %d. \n", flops_);	
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
	float * pCf, fdet;
	flops_ = 0;
	fsolveLU(pAf, nrowA, ncolA, & pCf);
	printf("Número de operações necessário para resolver o sistema: %d. \n", flops_);	
	free (pCf);
	flops_ = 0;
	fsolveLU(pAf, nrowA, ncolA, & pCf, & fdet);
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
	double * pAd = lermat("C", size, & nrowA, & ncolA);
	// Verifica se pode ser resolvido
	if (ncolA != nrowA + 1) {
		printf("O sistema não podem ser resolvido, porque as dimensões são incompatíveis: (%d x %d)! \n", nrowA, ncolA);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	// Resolve o sistema e relata o valor do determinante e o esforço computacional necessário para solução
	if (! fissym(pAf, nrowA, ncolA)) {
		printf("A matriz não é simétrica! \n");
		exit(13);
		}
	float fdet;
	flops_ = 0;
	float * pCf = fsolveChol(pAf, nrowA);
	if (pCf == NULL) {
		exit (6);
		}
	printf("Número de operações necessário para resolver o sistema: %d. \n", flops_);
	free (pCf);
	flops_ = 0;
	pCf = fsolveChol(pAf, nrowA, & fdet);
	if (pCf == NULL) {
		exit (6);
		}
	printf("Número de operações necessário para resolver o sistema com determinante: %d. \n", flops_);	
	flops_ = 0;
	printf("Determinante da matriz: 32 bits = %f. \n", fdet);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, NULL, NULL, nrowA, 1);
	return;
	}

void execprob6(int size) {
// Executa o problema número '6' com o tamanho 'size' indicado
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
	float * pBf = fmcopy(pBd, nrowB, ncolB);
	// Multiplica as matrizes por meio de rotinas diversas e compara o desempenho
	ucrono(true, 0);
	float * pCf = fmmult(pAf, nrowA, ncolA, pBf, nrowB, ncolB);
	ucrono(false, 1);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, NULL, NULL, nrowA, ncolB);
	ucrono(false, 0);
	for (int i = 0; i < 100; ++ i) {
		free(pCf);
		pCf = fgemm(pAf, nrowA, ncolA, pBf, nrowB, ncolB);
		}
	ucrono(false, 100);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, NULL, NULL, nrowA, ncolB);
	ucrono(false, 0);
	for (int i = 0; i < 100; ++ i) {
		free(pCf);
		pCf = fgemmref(pAf, nrowA, ncolA, pBf, nrowB, ncolB);
		}
	ucrono(false, 100);
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
	float * pCf = fsolveLS(pAf, nrowA, 1);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, NULL, NULL, nrowA, 1);
	return;
	}

void execprob8(int size) {
// Executa o problema número '8' com o tamanho 'size' indicado.
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
	// Resolve o sistema e relata o valor do determinante e o esforço computacional necessário para solução
	float fdet;
	flops_ = 0;
	float * pCf = fsolveDG(pAf, nrowA, & fdet);
	printf("Número de operações necessário para resolver o sistema: %d. \n", flops_);	
	flops_ = 0;
	printf("Determinante da matriz: 32 bits = %f \n", fdet);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, NULL, NULL, nrowA, 1);
	return;
	}

void execprob9(int size) {
// Executa o problema número '9' com o tamanho 'size' indicado.
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
	// Inverte a matriz e relata o valor do determinante
	float fdet;
	flops_ = 0;
	float * pInv = finvG(pAf, nrowA, ncolA, & fdet);
	printf("Número de operações necessário para inverter a matriz: %d. \n", flops_);	
	flops_ = 0;
	printf("Determinante da matriz: 32 bits = %f \n", fdet);
	// Obtém a solução
	float * pvet = (float *) malloc(nrowA * sizeof(float));
	if (pvet == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 1! \n", nrowA);
		exit(7);
		}
	for (int i = 0; i < nrowA; ++ i) {
		pvet[i] = pAf[i * ncolA + nrowA];
		}
	float * pCf = fmmult(pInv, nrowA, nrowA, pvet, nrowA, 1);
	printf("Número de operações necessário para resolver o sistema: %d. \n", flops_);	
	// Calcula e relata a norma 2 dos resultados
	calcn2(pCf, NULL, NULL, nrowA, 1);
	return;
	}
	
void execprob10(int size) {
// Executa o problema número '10' com o tamanho 'size' indicado.
	// Lê o sistema de entrada
	extern int flops_;
	int nrowA, ncolA;
	double * pAd = lermat("S", size, & nrowA, & ncolA);
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	// Relata a equação característica segundo os algoritmos implementados
	flops_ = 0;
	float * peq = feqcaracL(pAf, nrowA, ncolA);
	printf("Número de operações necessário para calcular a equação característica da matriz: %d. \n", flops_);	
	fshowmat(peq, nrowA + 1, 1, "Coeficientes da equação característica da matriz:");
	free(peq);
	float fdet, * pinv;
	flops_ = 0;
	peq = feqcaracLF(pAf, nrowA, ncolA, & fdet, & pinv);
	printf("Número de operações necessário para calcular a equação característica da matriz, determinante e inversa: %d. \n", flops_);	
	fshowmat(peq, nrowA + 1, 1, "Coeficientes da equação característica da matriz:");
	printf("Determinante: %f \n", fdet);
	if (fdet != 0) {
		fshowmat(pinv, nrowA, nrowA, "Matriz inversa");	
		}
	return;
	}

void execprob11(int size) {
// Executa o problema número '11' com o tamanho 'size' indicado.
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
	// Resolve o sistema por decomposição LU com refinamento
	flops_ = 0;
	int niter;
	float * pX, error;
	int retcode = fsolveLU(pAf, nrowA, ncolA, & pX, NULL, & niter, & error);
	printf("Número de operações necessário: %d. Número de iterações: %d. Erro: %f \n", flops_, niter, error);
	// Calcula e relata a norma 2 dos resultados
	calcn2(pX, NULL, NULL, nrowA, 1);
	return;
	}

void execprob12(int size) {
// Executa o problema número '12' com o tamanho 'size' indicado.
	// Lê o sistema de entrada
	extern int flops_;
	int nrowA, ncolA;
	double * pAd = lermat("D", size, & nrowA, & ncolA);
	// Verifica se pode ser resolvido
	if (ncolA != nrowA + 1) {
		printf("O sistema não podem ser resolvido, porque as dimensões são incompatíveis: (%d x %d)! \n", nrowA, ncolA);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	// Resolve pelo método iterativo de Jacobi
	float * pX;
	int niter;
	flops_ = 0;
	int retcode = fsolveJ(pAf, nrowA, & pX, & niter);
	printf("Número de operações necessário para resolver o sistema: %d. Iterações: %d. \n", flops_, niter);	
	flops_ = 0;
	// Calcula e relata a norma 2 dos resultados
	calcn2(pX, NULL, NULL, nrowA, 1);
	return;
	}

void execprob13(int size) {
// Executa o problema número '13' com o tamanho 'size' indicado.
	// Lê o sistema de entrada
	extern int flops_;
	int nrowA, ncolA;
	double * pAd = lermat("D", size, & nrowA, & ncolA);
	// Verifica se pode ser resolvido
	if (ncolA != nrowA + 1) {
		printf("O sistema não podem ser resolvido, porque as dimensões são incompatíveis: (%d x %d)! \n", nrowA, ncolA);
		exit(5);
		}
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	// Resolve pelo método iterativo de Gauss-Seidel
	float * pX;
	int niter;
	flops_ = 0;
	int retcode = fsolveGS(pAf, nrowA, & pX, & niter);
	// Resolve pelo método de Gauss-Seidel	
	printf("Número de operações necessário para resolver o sistema: %d. Iterações: %d. \n", flops_, niter);	
	flops_ = 0;
	// Calcula e relata a norma 2 dos resultados
	calcn2(pX, NULL, NULL, nrowA, 1);
	return;
	}

void execprob14(int size) {
// Executa o problema número '14' com o tamanho 'size' indicado.
	// Lê as matrizes de entrada
	extern int flops_;	
	int nrowA, ncolA, nrowB, ncolB;
	double * pAd = lermat("MatA", size, & nrowA, & ncolA);
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	// Encontra os autovalores extremos pelo método das potências e relata o esforço computacional necessário
	float maxav, minav = 0;
	int niter;
	flops_ = 0;
	int retcode = fmavP(pAf, nrowA, ncolA, & maxav , & minav, & niter);
	printf("Autovalores extremos: %f e %f. \n", minav, maxav);
	printf("Número de operações para cálculo dos autovalores extremos: %d. Iterações: %d. \n", flops_, niter);	
	return;
	}

void execprob15(int size) {
// Executa o problema número '15' com o tamanho 'size' indicado.
	// Lê as matrizes de entrada
	extern int flops_;	
	int nrowA, ncolA, nrowB, ncolB;
	double * pAd = lermat("C", size, & nrowA, & ncolA);
	// Cria versões em diversas precisões
	float * pAf = fmcopy(pAd, nrowA, ncolA);
	// Encontra os autovalores extremos pelo método de Jacobi e relata o esforço computacional necessário
	float * pav;
	int niter;
	flops_ = 0;
	int retcode = fmavJ(pAf, nrowA, ncolA, & pav , & niter);
	fshowmat(pav, nrowA, 1, "Autovalores");
	printf("Número de operações para cálculo dos autovalores extremos: %d. Iterações: %d. \n", flops_, niter);	
	return;
	}

	
// Funções para cálculo de autovalores por métodos iterativos
int fmavJ(float * pmat, int nrows, int ncols, float ** ppav, int * piter) {
// Calcula os autovalores extremos da matriz pelo método de Jacobi
	float maxofmaxes = 0;
	int retcode = 11, niter;
	for (niter = 0; niter < maxiter_; ++ niter) {
		for (int i = 0; i < nrows; ++ i) {
			float max = 0;
			int pos = -1;
			for (int j = 0; j < nrows; ++ j) {
				if (i == j) {
					continue;
					}
				float value = fabs(pmat[i * ncols + j]);
				if (value > max) {
					max = value;
					pos = j;
					}
				if (max > maxofmaxes) {
					maxofmaxes = max;
					}
				float aqq_app = pmat[pos * ncols + pos] - pmat[i * ncols + i];
				float apq = pmat[i * ncols + pos];
				float phi = 0.5 * aqq_app / apq;
				float t;
				if (phi == 0) {
					t = 1;
					}
				else {
					float root = sqrt(phi * phi + 1);
					float divisor;
					if (phi > 0) {
						divisor = phi + root;
						}
					else {
						divisor = phi - root;
						}
					t = 1 / divisor;
					}
				float t2 = t * t;
				float um_mais_t2 = 1 + t2;
				float cos2phi = 1 / um_mais_t2;
				float sinphicosphi = t * cos2phi;
				float cos2phi_sin2phi = (1 - t2) * cos2phi;;
				float newvalue = - aqq_app * sinphicosphi + apq * cos2phi_sin2phi;
				pmat[i * ncols + pos] = pmat[pos * ncols + i] = newvalue;
				}
			}
		if (maxofmaxes <= maxerr_) {
			retcode = 0;
			break;
			}
		}
	* piter = niter;
	float * pav = (float *) malloc (nrows * sizeof(float));
	if (pav == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 1! \n", nrows);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		pav[i] = pmat[i * ncols + i];
		}
	* ppav = pav;
	return retcode;
	}
	


int fmavP(float * pmat, int nrows, int ncols, float * pmax, float * pmin, int * piter) {
// Calcula os autovalores extremos da matriz pelo método das potências
	int niter1, niter2;
	int retcode = fmmaxavP(pmat, nrows, ncols, pmax, & niter1, true);
	retcode = fmmaxavP(pmat, nrows, ncols, pmin, & niter2, false);
	* piter = niter1 + niter2;
	return retcode;
	}
	
int fmmaxavP(float * pmat, int nrows, int ncols, float * pmax, int * piter, bool direto) {
	float * pY = (float *) malloc(nrows * sizeof(float));
	if (pY == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 1! \n", nrows);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		pY[i] = 1;
		}
	float * pL, * pU;
	int * pP;
	if (! direto) {
		f2LU(pmat, nrows, & pL, & pU, & pP, NULL);
		}
	float lastav = - 1e6, lasterror = 1e6, av, * pZ;
	int retcode = 11, niter;
	for (niter = 0; niter < maxiter_; ++ niter) {
		if (direto) {
			pZ = fmmult(pmat, nrows, ncols, pY, nrows, 1);
			}
		else {
			pZ = fdoLU(pY, pL, pU, pP, nrows);
			}
		float alpha = fmnormi(pZ, nrows, 1);
		float ynorm = fmnormi(pY, nrows, 1);
		av = alpha / ynorm;
		float error = fabs((av - lastav) / av);
		flops_ += 1 + 2 * FLOPS_DIV;
		if (debuglevel_ >= 1) {
			printf("Iter. %d: av = %f, error = %f \n", niter, av, error);
			}
		if (error <= maxerr_) {
			retcode = 0;
			break;
			}
		if (error > lasterror) {
			retcode = 12;
			break;
			}
		lastav = av;
		lasterror = error;
		for (int i = 0; i < nrows; ++ i) {
			pY[i] = pZ[i] / alpha;
			flops_ += FLOPS_DIV;
			}
		free(pZ);
		}
	if (retcode == 12) {
		printf("O método divergiu! \n");
		}
	free(pZ);
	* pmax = direto ? av : (1 / av);
	flops_ += FLOPS_DIV;
	* piter = niter;
	return retcode;
	}

// Funções para solução de sistemas por métodos iterativos
int fiterGS(float * pA, float * pB, float * pmerror, float * pcorr, float ** ppX, float * pM, float * pC, int * pim, int nrows) {
	float * pX = * ppX;
	for (int i = 0; i < nrows; ++ i) {
		float sum = 0;
		for (int j = 0; j < nrows; ++ j) {
			float x = pX[j];
			float coef = pA[i * nrows + j];
			if (i == j || x == 0 || coef == 0) {
				continue;
				}
			sum += coef * x;
			flops_ += 2;
			}
		float value = (pB[i] - sum) * pA[i * nrows + i];
		pcorr[i] = pX[i] - value;
		pX[i] = value;
		flops_ += 3;
		}
	return 0;
	}
	
int fiterJ(float * pA, float * pB, float * pmerror, float * pcorr, float ** ppX, float * pM, float * pC, int * pim, int nrows) {
	float * pX = * ppX;
	float * result = (float *) malloc(nrows * sizeof(float));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 1! \n", nrows);
		exit(7);
		}	
	for (int i = 0; i < nrows; ++ i) {
		float sum = 0;
		for (int j = 0; j < nrows; ++ j) {
			float x = pX[j];
			float coef = pA[i * nrows + j];
			if (i == j || x == 0 || coef == 0) {
				continue;
				}
			sum += coef * x;
			flops_ += 2;
			}
		if (prec_ == 0) {
			result[i] = (pB[i] - sum) * pA[i * nrows + i];
			flops_ += 2;
			}
		if (prec_ == 1) {
			result[i] = (pB[i] - sum);			
			++ flops_;
			}
		pcorr[i] = result[i] - pX[i];
		++ flops_;
		}
	* ppX = result;
	return 0;
	}

int fiterate(int type, f_iter * pfn, float * pA, float * pB, float ** ppX, int * piter, float * perror, float * pfm1, float * pfm2, int * pim, int nrows) {
	int niter, retcode = 11;
	float error, lastcorr = 1e6, lasterror = 1e6;
	float * fmerror, * corr = NULL, * lastpX = NULL, * result = * ppX;
	if (type == 0) {
		corr = (float *) malloc(nrows * sizeof(float));
		if (corr == NULL) {
			printf("Não conseguiu alocar memória para a matriz %d x 1! \n", nrows);
		exit(7);
			}
		}
	for (niter = 0; niter < maxiter_; ++ niter) {
		// Calcula o erro
		if (type == 1) {
			float * pAX = fmmult(pA, nrows, nrows, result, nrows, 1);
			fmerror = fmadd(pB, nrows, 1, pAX, nrows, 1, false);
			if (debuglevel_ >= 2) {
			fshowmat(fmerror, nrows, 1, "Erro");
			}
			free(pAX);
			error = fmnormi(fmerror, nrows, 1);
			if (debuglevel_ >= 1) {
			printf("Erro: %f \n", error);
			}
			if (error <= maxerr_) {
			retcode = 0;
			break;
			}
			if (error > lasterror) {
			retcode = 12;
			break;
			}
			}
		// Corrige e tenta de novo
		retcode = (* pfn)(pA, pB, fmerror, corr, & result, pfm1, pfm2, pim, nrows);
		if (retcode != 0) {
			break;
			}
		if (type == 0) {
			float ncorr = fmnormi(corr, nrows, 1);
			if (debuglevel_ >= 1) {
				printf("Correção: %f \n", ncorr);
				}
			if (ncorr <= maxerr_) {
				retcode = 0;
				break;
				}
			if (ncorr > lastcorr) {
				retcode = 12;
				break;
				}
			lastcorr = ncorr;				
			}
		if (type == 1) {
			lasterror = error;
			free(fmerror);
			}
		if (lastpX != NULL && lastpX != result) {
			free(lastpX);
			}
		lastpX = result;
		if (debuglevel_ >= 2) {
			printf("Iteração %d: \n", niter);
			fshowmat(result, nrows, 1, "X");
			}
		}
	if (retcode == 12) {
		if (lastpX != NULL && lastpX != result) {
			free(result);
			result = lastpX;
			}
		error = lasterror;
		printf("O método divergiu! \n");		
		}
	else {
		if (lastpX != NULL && lastpX != result) {
			free(lastpX);
			}
		}	
	if (debuglevel_ >= 2) {
		fshowmat(result, nrows, 1, "X");
		}
	* ppX = result;
	if (perror != NULL) {
		* perror = error;
		}
	if (piter != NULL) {
		* piter = niter;
		}
	if (debuglevel_ >= 1) {
		fshowmat(result, nrows, 1, "Result");
		}
	if (type == 0) {
		free(corr);
		}
	return retcode;
	}

int fsolveGS(float * psrc, int rank, float ** ppX, int * piter) {
// Resolve o sistema de equações pelo método de Gauss-Seidel
	extern int debuglevel_;
	int ncols = rank + 1;
	float * pA = (float *) malloc(rank * rank * sizeof(float));
	float * pB = (float *) malloc(rank * sizeof(float));
	float * pX = (float *) malloc(rank * sizeof(float));
	if (pA == NULL || pB == NULL || pX == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank + 2);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			float coef = psrc[i * ncols + j];
			if (i == j) {
				if (coef == 0) {
					printf("A matriz é singular! \n");
					exit(8);
					}
				float value = psrc[i * ncols + rank];
				pB[i] = value;
				coef = 1 / coef;
				pX[i] = value * coef;
				flops_ += 1 + FLOPS_DIV;
				}
			pA[i * rank + j] = coef;
			}
		}
	if (debuglevel_ >= 2) {
		fshowmat(pA, rank, rank, "A");
		fshowmat(pB, rank, 1, "B");
		fshowmat(pX, rank, 1, "X");
		}
	int retcode = fiterate(0, & fiterGS, pA, pB, & pX, piter, NULL, NULL, NULL, NULL, rank);
	* ppX = pX;
	return retcode;
	}
	
int fsolveJ(float * psrc, int rank, float ** ppX, int * piter) {
// Resolve o sistema de equações pelo método de Jacobi	
	extern int debuglevel_;
	int ncols = rank + 1;
	float * pA = (float *) malloc(rank * rank * sizeof(float));
	float * pB = (float *) malloc(rank * sizeof(float));
	float * pX = (float *) malloc(rank * sizeof(float));
	if (pA == NULL || pB == NULL || pX == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank + 2);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pA[i * rank + j] = psrc[i * ncols + j];
			}
		pB[i] = psrc[i * ncols + rank];
		}
	if (prec_ == 1) {
		float * pP = (float *) calloc(rank * rank, sizeof(float));
		if (pP == NULL) {
			printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank);
			exit(7);
			}	
		for (int i = 0; i < rank; ++ i) {
			pP[i * rank + i] = 1 / psrc[i * ncols + i];
			flops_ += FLOPS_DIV;
			}
		float * paux = fmmult(pP, rank, rank, pA, rank, rank);
		free(pA);
		pA = paux;
		paux = fmmult(pP, rank, rank, pB, rank, 1);
		free(pB);
		pB = paux;
		}
	for (int i = 0; i < rank; ++ i) {
		if (prec_ == 0) {
			float coef = pA[i * rank + i];
			coef = 1 / coef;
			pA[i * rank + i] = coef;
			pX[i] = pB[i] * coef;
			flops_ += 1 + FLOPS_DIV;
			}
		if (prec_ == 1) {
			pX[i] = pB[i];
			}
		}
	if (debuglevel_ >= 2) {
		fshowmat(pA, rank, rank, "A");
		fshowmat(pB, rank, 1, "B");
		fshowmat(pX, rank, 1, "X");
		}
	int retcode = fiterate(0, & fiterJ, pA, pB, & pX, piter, NULL, NULL, NULL, NULL, rank);
	* ppX = pX;
	return retcode;
	}


// Wrappers para funções da biblioteca Openblas
float * fgemm(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB) {
	float * pC = (float *) malloc(nrowA * ncolB * sizeof(float));
	if (pC == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrowA, ncolB);
		exit(7);
		}
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nrowA, ncolB, ncolA, 1, pA, ncolA, pB, ncolB, 0, pC, ncolB);
	return pC;
	}


// Wrappers para funções da biblioteca de referência (em Fortran)
extern"C" { void sgemm_(char *, char *, int *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *); }
float * fgemmref(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB) {
	extern int debuglevel_;	
	float * pC = (float *) malloc(nrowA * ncolB * sizeof(float));
	if (pC == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrowA, ncolB);
		exit(7);
		}
	char modea, modeb;
	modeb = modea = 'T';
	int m = nrowA, n = ncolB, k = ncolA, lda = ncolA, ldb = ncolB, ldc = ncolB;
	float alpha = 1, beta = 0;
	if (debuglevel_ == 2) {
		fshowmat(pA, nrowA, ncolA, "A");
		fshowmat(pB, nrowA, ncolB, "B");
		}
	sgemm_(&modea, &modeb, &m, &n, &k, &alpha, pA, &lda, pB, &ldb, &beta, pC, &ldc);
	return pC;
	}

	
// Wrappers para funções da biblioteca LAPACKE
float * fsolveLS(float * psys, int rank, int nrhs) {
	extern int debuglevel_;
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
	lapack_int retcode = LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', rank, rank, nrhs, pA, rank, pB, nrhs);
	if (retcode != 0) {
		printf("Função LAPACKE_sgels retornou erro no argumento %d", -retcode);
		exit(10);
		}
	if (debuglevel_ == 2) {
		fshowmat(pA, rank, nrhs, "X");
		}
	free(pA);
	return pB;
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

	
// Funções para multiplicação das matrizes em diversas precisões
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

float * fpower(float * pmat, int nrows, int ncols, int pot) {
// Retorna o resultado da potência 'pot' da matriz, com 'pot' >= 0
	extern int debuglevel_;
	if (pot == 0) {
		return fident(nrows);
		}
	float * pvals = (float *) malloc(nrows * nrows * sizeof(float));
	if (pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, nrows);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < nrows; ++ j) {
			pvals[i * nrows + j] = pmat[i * ncols + j];
			}
		}
	float * plast = pvals;
	for (int i = 2; i <= pot; ++ i) {
		float * result = fgemm(pvals, nrows, nrows, plast, nrows, nrows);
		if (debuglevel_ >= 2) {
			printf("A^%d", i);
			fshowmat(result, nrows, nrows, "");
			}
		if (i > 2) {
			free(plast);
			}
		plast = result;
		}
	return plast;
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
float * finvG(float * pmat, int rank, int ncols, float * pdet) {
// Retorna a matriz inversa obtida por Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	int ncolsSys = 2 * rank;
	float * pSys = (float *) calloc(rank * ncolsSys, sizeof(float));	
	if (pSys == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, ncolsSys);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pSys[i * ncolsSys + j] = pmat[i * ncols + j];
			}
		pSys[i * ncolsSys + rank + i] = 1;
		}
	if (debuglevel_ >= 2) {
		fshowmat(pmat, rank, ncols, "pmat");
		fshowmat(pSys, rank, ncolsSys, "pSys");
		}
	float * pTS = f2tri(pSys, rank, ncolsSys, pdet);
	if (debuglevel_ >= 2) {
		fshowmat(pTS, rank, ncolsSys, "pTS");
		}
	free(pSys);
	pSys = (float *) calloc(rank * (rank + 1), sizeof(float));
	if (pSys == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank + 1);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pSys[i * (rank + 1) + j] = pTS[i * ncolsSys + j];
			}
		}
	float * pInv = (float *) malloc(rank * rank * sizeof(float));
	if (pInv == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank);
		exit(7);
		}
	for (int j = 0; j < rank; ++ j) {
		for (int i = 0; i < rank; ++ i) {
			pSys[i * (rank + 1) + rank] = pTS[i * ncolsSys + rank + j];
			}
		if (debuglevel_ >= 2) {
			fshowmat(pSys, rank, rank + 1, "pSys");
			}
		float * result = fmtrisolve(pSys, rank, rank + 1, true);
		if (debuglevel_ >= 2) {
			fshowmat(result, rank, 1, "Resultado");
			}
		for (int i = 0; i < rank; ++ i) {
			pInv[i * rank  + j] = result[i];
			}
		free(result);
		}
	return pInv;
	}

float * fsolveG(float * psrc, int rank, float * pdet) {
// Retorna a solução do sistema por Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	float * pTS = f2tri(psrc, rank, rank + 1, pdet);
	float * result = fmtrisolve(pTS, rank, rank + 1, true);
	if (debuglevel_ >= 2) {
		fshowmat(result, rank, 1, "Resultado");
		}
	return result;
	}

float * f2tri(float * psrc, int rank, int ncols, float * pdet) {
// Retorna o resultado da Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	float * pval = (float *) calloc(rank * ncols, sizeof(float));
	if (pval == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, ncols);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < ncols; ++ j) {
			pval[i * ncols + j] = psrc[i * ncols + j];
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
		float invmaxval = 1 / maxval;
		flops_ += FLOPS_DIV;
		for (int i = j + 1; i < rank; ++ i) {
			double multiplier = pval[i * ncols + j] * invmaxval;
			++ flops_;
			pval[i * ncols + j] = 0;
			for (int k = j + 1; k < ncols; ++ k) {
				pval[i * ncols + k] -= pval[j * ncols + k] * multiplier;
				flops_ += 2;				
				}
			}
		if (debuglevel_ >= 2) {
			printf("Coluna %d eliminação. Pivô = %f\n", j, maxval);
			fshowmat(pval, rank, ncols, "");
			}
		}
	if (pdet != NULL) {
		for (int i = 0; i < rank; ++ i) {
			det *= pval[i * ncols + i];
			++ flops_;		
			}
		* pdet = det * (sinal ? -1 : 1);
		}
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
		double invmaxval = 1 / maxval;
		flops_ += FLOPS_DIV;
		for (int i = j + 1; i < rank; ++ i) {
			double multiplier = pval[i * ncols + j] * invmaxval;
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
	if (pdet != NULL) {
		for (int i = 0; i < rank; ++ i) {
			det *= pval[i * ncols + i];
			++ flops_;		
			}
		* pdet = det * (sinal ? -1 : 1);
		}
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
		double invmaxval = 1 / maxval;
		flops_ += FLOPS_DIV;
		for (int i = j + 1; i < rank; ++ i) {
			long double multiplier = pval[i * ncols + j] * invmaxval;
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
	if (pdet != NULL) {
		for (int i = 0; i < rank; ++ i) {
			det *= pval[i * ncols + i];
			++ flops_;		
			}
		* pdet = det * (sinal ? -1 : 1);
		}
	return pval;
	}

// ... Decomposição LU
float * finvChol(float * pmat, int rank, int ncols, float * pdet) {
// Retorna a matriz inversa obtida por Decomposição de Cholesky, em precisão simples.
	extern int debuglevel_;
	int ncolsSys = 2 * rank;
	float * pSys = (float *) calloc(rank * ncolsSys, sizeof(float));	
	if (pSys == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, ncolsSys);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pSys[i * ncolsSys + j] = pmat[i * ncols + j];
			}
		pSys[i * ncolsSys + rank + i] = 1;
		}
	if (debuglevel_ >= 2) {
		fshowmat(pmat, rank, ncols, "pmat");
		fshowmat(pSys, rank, ncolsSys, "pSys");
		}
	float * pTS = f2tri(pSys, rank, ncolsSys, pdet);
	if (debuglevel_ >= 2) {
		fshowmat(pTS, rank, ncolsSys, "pTS");
		}
	free(pSys);
	pSys = (float *) calloc(rank * (rank + 1), sizeof(float));
	if (pSys == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank + 1);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pSys[i * (rank + 1) + j] = pTS[i * ncolsSys + j];
			}
		}
	float * pInv = (float *) malloc(rank * rank * sizeof(float));
	if (pInv == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank);
		exit(7);
		}
	debuglevel_ = 2;
	for (int j = 0; j < rank; ++ j) {
		for (int i = 0; i < rank; ++ i) {
			pSys[i * (rank + 1) + rank] = pTS[i * ncolsSys + rank + j];
			}
		if (debuglevel_ >= 2) {
			fshowmat(pSys, rank, rank + 1, "pSys");
			}
		float * result = fmtrisolve(pSys, rank, rank + 1, true);
		if (debuglevel_ >= 2) {
			fshowmat(result, rank, 1, "Resultado");
			}
		for (int i = 0; i < rank; ++ i) {
			pInv[i * rank  + j] = result[i];
			}
		free(result);
		}
	debuglevel_ = 0;
	return pInv;
	}

int fmakeLU(float * pmat, int nrows, int ncols, float * values, int * position) {
// Carrega as componentes L ou U da matriz com os valores 'values'.
	extern int debuglevel_;
	for (int i = 0; i < nrows; ++ i) {
		int pos = (position == NULL) ? i : position[i]; 
		pmat[i * ncols + nrows] = values[pos];
		}
	if (debuglevel_ >= 2) {
		fshowmat(pmat, nrows, ncols, "Matriz L");
		}
	return 0;
	}
	
int fsolveLU(float * psys, int nrows, int ncols, float ** ppX, float * pdet, int * piter, float * perror) {
// Calcula a solução do sistema por decomposição LU com refinamentos sucessivos.
// Retorna 0 se tiver sucesso e um código de erro em caso contrário.
// Indica a solução, o erro e o número de iterações necessário.
	// Decomposição LU
	extern int debuglevel_;
	float * pL, * pU;
	int * pP;
	f2LU(psys, nrows, & pL, & pU, & pP, pdet);
	float * pA = (float *) malloc(nrows * nrows * sizeof(float));
	float * pB = (float *) malloc(nrows * sizeof(float));
	if (pA == NULL || pB == NULL) {
		printf("Não conseguiu alocar memória para as matrizes %d x %d! \n", nrows, nrows + 1);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < nrows; ++ j) {
			pA [i * nrows + j] = psys[i * ncols + j];
			}
		pB [i] = psys[i * ncols + nrows]; 
		}
	// Resolve o sistema
	float * result = fdoLU(pB, pL, pU, pP, nrows);
	int retcode = 0;
	// Refina a solução, se desejado
	if (piter != NULL && perror != NULL) {
		retcode = fiterate(1, & fiterLU, pA, pB, & result, piter, perror, pL, pU, pP, nrows);
		}
	* ppX = result;
	if (debuglevel_ >= 2) {
		fshowmat(result, nrows, 1, "Resultado");
		}
	return retcode;
	}
	
float * fdoLU(float * pB, float * pL, float * pU, int * pP, int nrows) {
// Resolve um sistema já preparado por meio de decomposição LU.
	fmakeLU(pL, nrows, nrows + 1, pB, pP);
	float * pval = fmtrisolve(pL, nrows, nrows + 1, false);
	for (int i = 0; i < nrows; ++ i) {
		pU[(i + 1) * (nrows + 1) - 1] = pval[i]; 
		}
	free(pval);
	return fmtrisolve(pU, nrows, nrows + 1, true);
	}

int fiterLU(float * pA, float * pB, float * pmerror, float * pcorr, float ** ppX, float * pL, float * pU, int * pP, int nrows) {	
	float * pX = * ppX;
	float * corr = fdoLU(pmerror, pL, pU, pP, nrows);
	if (debuglevel_ >= 2) {
		fshowmat(corr, nrows, 1, "Correção");
		}
	float mcorr = fmnormi(corr, nrows, 1);
	float mX = fmnormi(pX, nrows, 1);
	float finc = mcorr/mX;
	++ flops_;
	if (debuglevel_ >= 1) {
		printf("Correção: %f \n", finc);
		}
	if (finc <= maxerr_) {
		return 12;
		}	
	float * result = fmadd(pX, nrows, 1, corr, nrows, 1, true);
	free(corr);
	* ppX = result;
	return 0;
	}
	
void f2LU(float * psrc, int rank, float ** ppL, float ** ppU, int ** ppP, float * pdet){
// Calcula o resultado da decomposição LU e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;		
	int ncols = 2 * rank + 1;
	float * pval = (float *) calloc(rank * ncols, sizeof(float));
	float * pL = (float *) calloc(rank * (rank + 1), sizeof(float));
	float * pU = (float *) calloc(rank * (rank + 1), sizeof(float));
	int * pP = (int *) malloc(rank * sizeof(int));
	if (pval == NULL || pL == NULL || pU == NULL || pP == NULL) {	
		printf("Não conseguiu alocar memória para as matrizes %d x %d! \n", rank, ncols + 2);
		exit(7);
		}
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
		float invmaxval = 1 / maxval;
		flops_ += FLOPS_DIV;
		for (int i = j + 1; i < rank; ++ i) {
			double multiplier = pval[i * ncols + j] * invmaxval;
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
		if (pdet != NULL) {
			det *= pval[i * ncols + i];
			++ flops_;
			}
		pval[i * ncols + rank + i] = 1;
		}
	if (pdet != NULL) {
		* pdet = det * (sinal ? -1 : 1);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pU[i * (rank + 1) + j] = pval[i * ncols + j];
			}
		for (int j = 0; j < rank; ++ j) {
			pL[i * (rank + 1) + j] = pval[i * ncols + j + rank];
			}
		pP[i] = pval[(i + 1) * ncols - 1];
		}
	* ppL = pL;
	* ppU = pU;
	* ppP = pP;
	return;
	}
	
// ... Decomposição de Cholesky
float * fsolveChol(float * psrc, int rank, float * pdet) {
// Retorna a solução do sistema por decomposição de Cholesky e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	float * pL = f2Chol(psrc, rank, pdet);
	if (pL == NULL) {
		return NULL;
		}
	float * pU = ftranspose(pL, rank, rank + 1);
	float * pB = (float *) malloc(rank * sizeof(float));
	if (pB == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 1! \n", rank);
		exit(7);
		}
	if (debuglevel_ >= 2) {
		fshowmat(pU, rank, rank + 1, "Transposta");
		}
	for (int i = 0; i < rank; ++ i) {
		pB[i] = psrc[i * (rank + 1) + rank]; 
		}
	float * result = fdoLU(pB, pL, pU, NULL, rank);
	if (debuglevel_ >= 2) {
		fshowmat(result, rank, 1, "Resultado");
		}
	return result;
	}

float * f2Chol(float * psrc, int rank, float * pdet) {
// Retorna o resultado da decomposição de Cholesky e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	int ncols = rank + 1;
	float * pdiv = (float *) malloc(rank * sizeof(float));
	float * pval = (float *) calloc(rank * ncols, sizeof(float));
	if (pval == NULL) {
		printf("Não conseguiu alocar memória para a matriz estendida %d x %d! \n", rank, ncols);
		exit(7);
		}
	float det = 1;
	for (int i = 0; i < rank; ++ i) {
		float sum;
		for (int j = 0; j <= i; ++ j) {
			sum = 0;
            for (int k = 0; k < j; ++ k) {
				sum += pval[i * ncols + k] * pval[j * ncols + k];
				flops_ += 2;
				}
            if (i == j) {
				float quadrado = psrc[i * (rank + 2)] - sum;
				if (quadrado <= 0) {
					printf("A matriz não é definida positiva!");
					return NULL;
					}
				float valor = sqrt(quadrado);
				pval[i * ncols + i] = valor;
				pdiv[i] = 1 / valor;
				flops_ += 1 + FLOPS_SQRT + FLOPS_DIV;
				if (pdet != NULL) {
					det *= valor;
					++ flops_;
					}
				if (debuglevel_ >= 3) {
					printf("L%d%d = %f \t", i, j, valor);
					}				
				}
			else {
				float valor = (psrc[i * (rank + 1) + j] - sum) * pdiv[j];
				pval[i * ncols + j] = valor;
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
		fshowmat(pval, rank, ncols, "Decomposição de Cholesky");
		}
	if (pdet != NULL) {
		* pdet = det * det;
		++ flops_;
		}
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
			pdst[i * ncols + j] = psrc[j * ncols + i];
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
float * finvTS(float * pmat, int rank, int ncols, bool superior) {
// Retorna a matriz inversa obtida por Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão simples.
	extern int debuglevel_;
	int ncolsSys = rank + 1;
	float * pSys = (float *) malloc(rank * ncolsSys * sizeof(float));	
	if (pSys == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, ncolsSys);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		for (int j = 0; j < rank; ++ j) {
			pSys[i * ncolsSys + j] = pmat[i * ncols + j];
			}
		}
	if (debuglevel_ >= 2) {
		fshowmat(pmat, rank, ncols, "pmat");
		fshowmat(pSys, rank, ncolsSys, "pSys");
		}
	float * pInv = (float *) malloc(rank * rank * sizeof(float));
	if (pInv == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank);
		exit(7);
		}
	for (int j = 0; j < rank; ++ j) {
		for (int i = 0; i < rank; ++ i) {
			pSys[i * ncolsSys + rank] = (i == j) ? 1 : 0;
			}
		if (debuglevel_ >= 2) {
			fshowmat(pSys, rank, ncolsSys, "pSys");
			}
		float * result = fmtrisolve(pSys, rank, ncolsSys, superior);
		if (debuglevel_ >= 2) {
			fshowmat(result, rank, 1, "Resultado");
			}
		for (int i = 0; i < rank; ++ i) {
			pInv[i * rank + j] = result[i];
			}
		free(result);
		}
	if (debuglevel_ >= 2) {
		fshowmat(pInv, rank, rank, "Inversa");
		}
	return pInv;
	}

float * fmtrisolve(float * pmat, int nrows, int ncols, bool superior) {
	float * result;
	result = (float *) malloc(nrows * sizeof(float));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		exit(7);
		}
	if (superior) {
		for (int i = nrows - 1; i >= 0; -- i) {
			float divisor = pmat[i * ncols + i];
			if (divisor == 0) {
				printf("O sistema é singular! %d \n");
				exit(8);
				}
			float invdivisor = 1 / divisor;
			flops_ += FLOPS_DIV;
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
			float parm = pmat[i * ncols + nrows];
			float valor = (parm - sum) * invdivisor;
			++ flops_;			
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %f = (%f - %f) / %f \n", i, valor, parm, sum, divisor);
				}
			}
		}
	else {
		for (int i = 0; i < nrows; ++ i) {
			float divisor = pmat[i * ncols + i];
			if (divisor == 0) {
				printf("O sistema é singular! \n");
				exit(8);
				}
			float invdivisor = 1 / divisor;
			flops_ += FLOPS_DIV;
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
			float parm = pmat[i * ncols + nrows];
			float valor = (parm - sum) * invdivisor;
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
			double invdivisor = 1 / divisor;
			flops_ += FLOPS_DIV;
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
			double valor = (parm - sum) * invdivisor;
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
			float invdivisor = 1 / divisor;
			flops_ += FLOPS_DIV;
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
			double valor = (parm - sum) * invdivisor;
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
			long double invdivisor = 1 / divisor;
			flops_ += FLOPS_DIV;
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
			long double valor = (parm - sum) * invdivisor;
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
			long double invdivisor = 1 / divisor;
			flops_ += FLOPS_DIV;
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
			long double valor = (parm - sum) * invdivisor;
			++ flops_;			
			result[i] = valor;
			if (debuglevel_ >= 2) {
				printf("Linha %d: %Lf = (%Lf - %Lf) / %Lf \n", i, valor, parm, sum, divisor);
				}
			}
		}
	return result;
	}

	
// Funções para cálculo das normas das matrizes em diversas precisões
float fmnormi(float * pmat, int nrow, int ncol) {
	if (pmat == NULL) {
		return 0;
		}
	int size = nrow * ncol;
	float max = 0;
	while (size -- > 0) {
		float valor = fabs(* pmat ++);
		max = (valor > max) ? valor : max;
		}
	return max;
	}

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
	

// Outras funções úteis
void ucrono(bool init, int divisor) {
// Marca os tempos transcorridos entre chamadas
	static double wstart, wend;
    static HANDLE hProcess;
    static FILETIME ftCreation, ftExit, ftKernel, ftUser1, ftUser2;
    static SYSTEMTIME stUser1,stUser2;
	if (init) {
		// Inicializa as variáveis
		hProcess = GetCurrentProcess();
		}
	if (divisor == 0) {
		// Começa a contagem
		GetProcessTimes(hProcess, &ftCreation, &ftExit, &ftKernel, &ftUser1);
		wstart = omp_get_wtime();
		}
	else {
		// Encerra a contagem e imprime o resultado
		GetProcessTimes(hProcess, &ftCreation, &ftExit, &ftKernel, &ftUser2);	
		wend = omp_get_wtime();
		FileTimeToSystemTime(& ftUser1, & stUser1);
		FileTimeToSystemTime(& ftUser2, & stUser2);
		double twall = (1000.0 * (wend - wstart)) / divisor;
		double tuser = (1000.0 * (stUser2.wSecond - stUser1.wSecond) + stUser2.wMilliseconds - stUser1.wMilliseconds) / divisor;
		printf("Tempo gasto = %f ms, user time = %f ms \n", twall, tuser);
		}
	return;
	}

float * feqcaracL(float * pmat, int nrows, int ncols) {
// Retorna os coeficientes do polinômio característico da matriz 'pmat' usando algoritmo de Leverrier
	extern int flops_;
	int rank = nrows + 1;
	float * coef = (float *) malloc(rank * sizeof(float));
	float * traces = (float *) malloc(rank * sizeof(float));
	if (coef == NULL || traces == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 2! \n", rank);
		exit(7);
		}
	float * pvals = (float *) malloc(nrows * nrows * sizeof(float));
	if (pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, nrows);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < nrows; ++ j) {
			pvals[i * nrows + j] = pmat[i * ncols + j];
			}
		}
	float * plast = pvals;
	coef[0] = -1;
	for (int i = 1 ; i < rank ; ++ i) {
		if (debuglevel_ >= 3) {
			printf("Coef. %d: ", i);
			}
		if (i > 1) {
			float * result = fmmult(pvals, nrows, nrows, plast, nrows, nrows);
			// flops_ += pow(nrows, FLOPS_MMULT_EXP);
			if (debuglevel_ >= 2) {
				printf("A^%d \n", i);
				fshowmat(result, nrows, nrows, "");
				}
			if (i > 2) {
				free(plast);
				}
			plast = result;
			}
		traces[i] = ftrace(plast, nrows, ncols);
		if (debuglevel_ >= 3) {
			printf("s = %f, ", traces[i]);
			}
		float sum = 0;
		for (int k = 1 ; k < i ; ++ k) {
			sum += coef[k] * traces[i - k];
			flops_ += 2;
			}
		coef[i] = (traces[i] - sum) / i;
		flops_ += 1 + FLOPS_DIV;			
		if (debuglevel_ >= 3) {
			printf("sum = %f, a = %f: ", sum, coef[i]);
			}
		}
	return coef;
	}

float * feqcaracLF(float * pmat, int nrows, int ncols, float * pdet, float ** ppinv) {
// Retorna os coeficientes do polinômio característico da matriz 'pmat' usando algoritmo de Leverrier-Faddeev. Também calcula o determinante e a matriz inversa.
	extern int flops_;
	int rank = nrows + 1;
	float nrows2 = nrows * nrows;
	float * coef = (float *) malloc(rank * sizeof(float));
	if (coef == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 2! \n", rank);
		exit(7);
		}
	float * pvals = (float *) malloc(nrows2 * sizeof(float));
	if (pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, nrows);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < nrows; ++ j) {
			pvals[i * nrows + j] = pmat[i * ncols + j];
			}
		}
	float * plast = pvals, * pdiff;
	bool signal = true;
	coef[0] = -1;	
	for (int i = 1 ; i < rank; ++ i) {
		signal = ! signal;
		if (debuglevel_ >= 3) {
			printf("Coef. %d: ", i);
			}
		if (i > 1) {
			float * pident = fident(nrows, coef[i - 1]);
			pdiff = fmadd(plast, nrows, nrows, pident, nrows, nrows, false);
			free(pident);
			float * result = fmmult(pvals, nrows, nrows, pdiff, nrows, nrows);
			// float * result = fgemm(pvals, nrows, nrows, pdiff, nrows, nrows);
			// flops_ += pow(nrows, FLOPS_MMULT_EXP);
			if ((i != rank - 1) || (ppinv == NULL)) {
				free(pdiff);
				}
			if (debuglevel_ >= 2) {
				printf("A%d \n", i);
				fshowmat(result, nrows, nrows, "");
				}
			if (i > 2) {
				free(plast);
				}
			plast = result;
			}
		coef[i] = ftrace(plast, nrows, ncols) / i;
		flops_ += FLOPS_DIV;			
		if (debuglevel_ >= 3) {
			printf("q = %f: ", coef[i]);
			}
		}
	coef[rank] = plast[0];
	float fdet;
	if (pdet != NULL) {
		fdet = signal ? - coef[rank] : coef[rank];
		* pdet = fdet;
		}
	if ((ppinv != NULL) && (* pdet != 0)) {
		* ppinv = fmtimes(pdiff, rank, rank, 1/fdet);
		}
	return coef;
	}

void ffromsys(float * psys, int nrows, int ncols, float ** ppA, float ** ppB) {
	int nrows2 = nrows * nrows;
	float * pmat = (float *) malloc(nrows2 * sizeof(float));
	float * pvals = (float *) malloc(nrows * sizeof(float));
	if (pmat == NULL || pvals == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, nrows + 1);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < nrows; ++ j) {
			pmat[i * nrows + j] = psys[i * ncols + j];
			}
		pvals[i] = psys[i * ncols + nrows];
		}
	* ppA = pmat;
	* ppB = pvals;
	return;
	}
	
float * fident(int rank, float val) {
// Retorna uma matriz identidade com o 'rank' indicado
	float * result = (float *) calloc(rank * rank, sizeof(float));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", rank, rank);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		result[i * (rank + 1)] = val;
		}
	return result;
	}

bool fisddom(float * pmat, int nrows, int ncols) {
// Verifica se a matriz 'pmat' é diagonalmente dominante.
	for (int i = 0; i < nrows; ++ i) {
		float sum = 0, ref = pmat[i * (ncols + 1)];
		for (int j = 0; j < nrows; ++ j) {
			if (j != i) {
				sum += pmat[i * ncols + j];
				if (sum >= ref) {
					return false;
					}
				}
			}
		}
	return true;
	}

bool fissym(float * pmat, int nrows, int ncols) {
	for (int i = 1; i < nrows; ++ i) {
		for (int j = 0; j < i; ++ j) {
			if (pmat[i * ncols + j] != pmat[j * ncols + i]) {
				return false;
				}
			}
		}
	return true;
	}

float * fmadd(float * pA, int nrowA, int ncolA, float * pB, int nrowB, int ncolB, bool add) {
// Retorna a soma das matrizes indicadas
	extern int flops_;
	int nrows = (nrowA > nrowB) ? nrowA : nrowB;
	int ncols = (ncolA > ncolB) ? ncolA : ncolB;
	float * result = (float *) malloc(nrows * ncols * sizeof(float));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		exit(7);
		}
	for (int i = 0; i < nrows; ++ i) {
		for (int j = 0; j < ncols; ++ j) {
			if (add) {
				result[i * ncols + j] = pA[i * ncolA + j] + pB[i * ncolB + j];
				}
			else {
				result[i * ncols + j] = pA[i * ncolA + j] - pB[i * ncolB + j];
				}
			++ flops_;
			}
		}
	return result;
	}
	
float * fmtimes(float * pmat, int nrows, int ncols, float value) {
	int size = nrows * ncols;
	float * result = (float *) malloc(size * sizeof(float));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x %d! \n", nrows, ncols);
		exit(7);
		}
	for (int i = 0; i < size; ++ i) {
		result[i] = pmat[i] * value;
		++ flops_;
		}
	return result;
	}

float * f2diag(float * psrc, int rank, float * pdet) {
// Diagonaliza um sistema por meio da Eliminação Gaussiana com pivotação e informa o valor do determinante, em precisão simples.
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
	for (int j = 0; j < rank; ++ j) {
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
		float invmaxval = 1 / maxval;
		flops_ += FLOPS_DIV;			
		for (int i = j + 1; i < rank; ++ i) {
			double multiplier = pval[i * ncols + j] * invmaxval;
			++ flops_;
			pval[i * ncols + j] = 0;
			for (int k = j + 1; k <= rank; ++ k) {
				pval[i * ncols + k] -= pval[j * ncols + k] * multiplier;
				flops_ += 2;				
				}
			}
		for (int i = j - 1; i >= 0; -- i) {
			double multiplier = pval[i * ncols + j] * invmaxval;
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

float * fsolveDG(float * psrc, int rank, float * pdet) {
// Retorna a solução do sistema por diagonalização e informa o valor do determinante, em precisão simples.
	extern int debuglevel_, flops_;
	float * pD = f2diag(psrc, rank, pdet);
	float * result = (float *) malloc(rank * sizeof(float));
	if (result == NULL) {
		printf("Não conseguiu alocar memória para a matriz %d x 1! \n", rank);
		exit(7);
		}
	for (int i = 0; i < rank; ++ i) {
		result[i] = pD[i * (rank + 1) + rank] / pD[i * (rank +1) + i];
		flops_ += FLOPS_DIV;
		}		
	if (debuglevel_ >= 2) {
		fshowmat(result, rank, 1, "Resultado");
		}
	return result;
	}

float ftrace(float * pmat, int nrows, int ncols) {
// Retorna o traço da matriz 'pmat'.
	float sum = 0;
	for (int i = 0; i < nrows; ++ i) {
		sum += pmat[i * (ncols + 1)];
		}
	return sum;
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
