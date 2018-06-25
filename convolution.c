#include "stdio.h"
#include "omp.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"
static int const IM_size = 2000;
static int const K_size = 5;
static int const K_pix = 25;

void convolution(unsigned char *Im,unsigned char *MC,int Mask[][K_size]){
  int my_rank=omp_get_thread_num();
  int thread_count=omp_get_num_threads();
  int hx=IM_size/thread_count;
  int hxmin=(int)(hx*my_rank);
  int hxmax=(int)(hx*(my_rank+1));
  for (int px = hxmin; px < hxmax; px++) {
    for (int py = 0; py < IM_size; py++) {
      int acumv=0;
      int acumt=0;
      int lim_s=IM_size-2;
      if (px >= 2 && px < lim_s && py >= 2 && py < lim_s){
        for (int k1 = 0; k1 < K_size; k1++) {
          for (int k2 = 0; k2 < K_size; k2++) {
            int indx=px+k1-2;
            int indy=py+k2-2;
            acumv+=*(Im + indx*IM_size + indy)*Mask[k1][k2];
            acumt=K_pix;
          }
        }
      }
      else{
        for (int k1 = 0; k1 < K_size; k1++) {
          for (int k2 = 0; k2 < K_size; k2++) {
            int indx=px+k1-2;
            int indy=py+k2-2;
            if (indx >= 0 && indx < IM_size && indy >= 0 && indy < IM_size){
              acumv+=*(Im + indx*IM_size + indy)*Mask[k1][k2];
              acumt+=1;
            }
          }
        }
      }
      *(MC + px*IM_size + py)=(int)(acumv/acumt);
    }
  }
}

int main(int argc, char* argv[]) {
srand (time(NULL));
unsigned char *Im = (unsigned char *)malloc(IM_size * IM_size * sizeof(unsigned char));
unsigned char *MC = (unsigned char *)malloc(IM_size * IM_size * sizeof(unsigned char));
int thread_count=strtol(argv[1],NULL,10);
for (int i = 0; i < IM_size; i++) {
  for (int j = 0; j < IM_size; j++) {
    *(Im + i*IM_size + j)=rand()%100+1;
  }
}

printf("Generación aleatoria\n");

int Mask[K_size][K_size];
for (int i = 0; i < K_size; i++) {
  for (int j = 0; j < K_size; j++) {
    Mask[i][j]=1;
  }
}
printf("Generando la máscara\n");
double start = omp_get_wtime( );
#pragma omp parallel num_threads(thread_count)
convolution(Im,MC,Mask);
double end = omp_get_wtime( );
printf("time = %lf ms\n",(end-start));
//printf("start = %lf\nend = %lf\ntime = %lf ms\n",start, end, (end-start));
printf("Terminando convolucion\n");
return 0;
}
