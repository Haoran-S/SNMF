/*
 C code for our sBSUM algorithm (parallel version), to reproduce our works on SNMF research.
 References:
 [1] Qingjiang Shi, Haoran Sun, Songtao Lu, Mingyi Hong, and Meisam Razaviyayn. 
     "Inexact Block Coordinate Descent Methods For Symmetric Nonnegative Matrix Factorization." 
     arXiv preprint arXiv:1607.03092 (2016).
 
 version 1.0 -- April/2016
 Written by Haoran Sun (hrsun AT iastate.edu)
*/
   
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <mpi.h>
#define max(a,b) (a>b?a:b)
int MatMatProd(int m, int n, double *X, double *XXT);
int randperm(int n, int perm[]);
double ReadX(int m, int n, double *X);
double ReadX2(int m, int n, double *X);
double ReadlocalM(int size, int rank, int n, int nlocal, double *M);
double OBJpara(int m, int n, int nlocal, int kk, double *X, double *M);

int main(int argc, char **argv)
{
    FILE  *fp;
    int i,j,k,m,n,index,rank,size;
    double objective;
    double ElapsedTime;
    double t1,t2;
    int iter = 0;
    ElapsedTime = 0.0;

    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    fp = fopen("./data/dim.txt", "r");
    if (!fp)
    {
        printf("Unable to open file dim!");
        return 1;
    }
    if (fp)
    {
        fscanf (fp, "%d", &n);
        fscanf (fp, "%d", &m);
        if (rank==0)
        {
             printf("%d %d %d\n",m,n,size);
        }
    }
    fclose(fp);
    
    const int MAX_ITER=200;
    const int B_ITER=1;
    double objall[MAX_ITER], timeall[MAX_ITER];
    
    double *X;
    X = (double*) calloc(m*n,sizeof(double));
    ReadX(m, n, X);
    
    double *X2;
    X2 = (double*) calloc(m*n,sizeof(double));
    ReadX2(m, n, X2);
    
    int nlocal;
    int blocklen = (int) ceil((n+0.0)/size);
    if (rank < size-1)
        nlocal = blocklen;
    else
        nlocal = (n - rank*blocklen);
 
    double *localM;
    localM = (double*) calloc(n*nlocal,sizeof(double));
    ReadlocalM(size, rank,n,nlocal,localM);
    
    double *XXT;
    XXT = (double*)calloc(m*m,sizeof(double));
    MatMatProd(m, n, X, XXT);
    
    double *XTXd;
    XTXd = (double*)calloc(n,sizeof(double));
    for (i=0;i<n;i++)
    {
        for (k=0;k<m;k++)
        {
            XTXd[i] += X[k+i*m]*X[k+i*m];
        }
    }

    /* ------------------ main loop ------------------ */
    while (iter < MAX_ITER)
    {
        double *objsend;
        double *objrecv;
        objsend = (double*) calloc(1,sizeof(double));
        objrecv = (double*) calloc(1,sizeof(double));
        objsend[0]=OBJpara(m,n,nlocal,rank*blocklen,X,localM);;
        MPI_Reduce(objsend, objrecv, 1, MPI_DOUBLE,
                   MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank==0)
        {
            printf("%d %f %f\n",iter,objrecv[0],ElapsedTime);
        }
 
       
        t1=MPI_Wtime();
        int Rndint [m*nlocal];
        randperm(m*nlocal,Rndint);
        double a,b,c,d,p,q,x,diff,mind;double nRC;
        int k;double w, delta;float XM=0;float XXTX=0;

        for (index=0;index<m*nlocal;index++)
        {
            nRC = Rndint[index];
            j = floor(nRC/m);
            i = nRC-j*m;
            j=j+rank*blocklen;

            a = 4;
            b = 12*X[i+j*m];
            c = 4*(XXT[i*m+i]-localM[j+(j-rank*blocklen)*n]+XTXd[j]+X[i+j*m]*X[i+j*m]);
            
            XXTX=0;
            for (k=0;k<m;k++)
            {
                XXTX+=XXT[i*m+k]*X[k+j*m];
            }
            XM=0;
            for (k=0;k<n;k++)
            {
                XM+=X2[i*n+k]*localM[k+(j-rank*blocklen)*n];
            }
            
            d = 4*XXTX-4*XM;
            c = c+max(b*b/(3.0*a)-c,0);

            p = (3*a*c-b*b)/(3.0*a*a);
            q = (9*a*b*c-27*a*a*d-2*b*b*b)/(27.0*a*a*a);

            delta=q*q/4.0+p*p*p/27.0;
            if (c > (b*b/(3.0*a)))
            {
                w=cbrt(q/2.0-sqrt(delta))+cbrt(q/2.0+sqrt(delta));
            }
            else
            {
                w=cbrt(b*b*b/(27.0*a*a*a)-d/4.0);
            }
            x=max(w,0);
            
            double *SEND;
            double *RECV;
            SEND = (double*) calloc(3,sizeof(double));
            RECV = (double*) calloc(3*size,sizeof(double));

            SEND[0]=x;
            SEND[1]=i;
            SEND[2]=j;
            MPI_Allgather(SEND,3,MPI_DOUBLE,RECV,3,MPI_DOUBLE,MPI_COMM_WORLD);
            
            int kkk=0;
            for (kkk=0;kkk<size;kkk++)
            {
                x=RECV[kkk*3+0];
                i=RECV[kkk*3+1];
                j=RECV[kkk*3+2];

                XXT[i*m+i] = XXT[i*m+i]+(x-X[i+j*m])*(x-X[i+j*m]);
                for (k=0;k<m;k++)
                {
                    XXT[k*m+i] = XXT[k*m+i]+(x-X[i+j*m])*X[k+j*m];
                }
                for (k=0;k<m;k++)
                {
                    XXT[i*m+k] = XXT[i*m+k]+(x-X[i+j*m])*X[k+j*m];
                }
                XTXd[j]  = XTXd[j]+ (x-X[i+j*m])*(x+X[i+j*m]);
                X[i+j*m]=x;
                X2[i*n+j]=x;
            }
        }
        t2=MPI_Wtime();
        ElapsedTime = ElapsedTime +t2-t1;
        
        
        iter++;
    }

    
//    if (rank == 0)
//    {
//        fp = fopen("./data/recoverX.txt", "w");
//
//        for (j=0;j<n;j++)
//        {
//            for (i=0;i<m;i++)
//            {
//                fprintf (fp, "%f ", X[i+j*m]);
//            }
//            fprintf(fp,"\n");
//        }
//
//        fclose(fp);
//        printf("done.\n");
//    }

    MPI_Finalize();
    return 0;
}

/* ------------------ XXT ------------------ */

int MatMatProd(int m, int n, double *X, double *XXT)
{
    int i,j,k;
    for (i=0;i<m;i++)
    {
        for (j=0;j<m;j++)
        {
            for (k=0;k<n;k++)
            {
                XXT[i*m+j]+=X[i+k*m]*X[j+k*m];
            }
        }
    }
    return 0;
}

/* ------------------ random generate ------------------ */

int randperm(int n, int perm[])
{
    int i, j, t;
    
    for(i=0; i<n; i++)
        perm[i] = i;
    
    for(i=0; i<n; i++)
    {
        j = rand()%(n-i)+i;
        t = perm[j];
        perm[j] = perm[i];
        perm[i] = t;
    }
    return 0;
}

/* ------------------ Read File ------------------ */

double ReadX(int m, int n, double *X)
{
    FILE  *fp;
    int i,j;
    double v;
    fp = fopen("./data/dataX.txt", "r");
    if (fp)
    {
        
        for (j=0;j<n;j++)
        {
            for (i=0;i<m;i++)
            {
                fscanf (fp, "%lf", &v);
                X[i+j*m] = v;
                
            }
        }
    }
    fclose(fp);
    return 0;
}

double ReadX2(int m, int n, double *X)
{
    FILE  *fp;
    int i,j;
    double v;
    fp = fopen("./data/dataX.txt", "r");
    if (fp)
    {
        
        for (j=0;j<n;j++)
        {
            for (i=0;i<m;i++)
            {
                fscanf (fp, "%lf", &v);
                X[i*n+j] = v;
            }
        }
    }
    fclose(fp);
    return 0;
}

double ReadlocalM(int size, int rank, int n, int nlocal, double *M)
{
    FILE  *fp;
    int i,j;
    double v;
    char s[40];
    sprintf(s, "./data/%d/dataM%d.txt", size,rank);
    fp = fopen(s, "r");
    if (fp)
    {
        for (i=0;i<n;i++)
        {
            for (j=0;j<nlocal;j++)
            {
                fscanf (fp, "%lf", &v);
                M[i+j*n] = v;
            }
        }
    }
    fclose(fp);
    return 0;
}




double OBJpara(int m, int n, int nlocal, int kk, double *X, double *M)
{
    int i,j,k;
    double objective = 0.0;
    double temp;
    
    for (i=0;i<n;i++)
    {
        for (j=0;j<nlocal;j++)
        {
            temp=0;
            for (k=0;k<m;k++)
            {
                temp+=X[k+(j+kk)*m]*X[k+i*m];
            }
            temp=temp-M[i+n*j];
            objective+= temp*temp;
        }
    }
    
    return objective;
}








