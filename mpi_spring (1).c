#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <time.h>
#include <mpi.h>

//********************************************
//  This structure defines the data required
//  to model a spring
//********************************************
struct SpringODE {
   int numEqns;
   double s;
   double q[2];
   double mass;
   double k;
   double mu;
};

    int rank, comm_size;

//***********************
//  Function prototypes
//***********************
void springRightHandSide( struct SpringODE *spring, double *q, 
             double *deltaQ, double ds, double qScale, double *dq);
void springRungeKutta4(struct SpringODE *spring, double ds);
void springRK2(struct SpringODE *spring, double ds);

//******************************************************
//  Main method. It initializes a spring and solves
//  for the spring motion using the Runge-Kutta solver
//******************************************************
int main(int argc, char *argv[]) {
  double total_time;
  clock_t start, end;
  start = clock();

  struct SpringODE spring;
  double dt = 0.1;
  int i;

  //  Initialize spring
  spring.mass = 1.0;
  spring.mu = 1.5;
  spring.k = 20.0;
  spring.numEqns = 2;
  spring.s = 0.0;      //  time = 0.0
  spring.q[0] = 0.0;   //  vx = 0.0
  spring.q[1] = -0.2;  //  x = -0.2
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  //  Solve for the spring motion over a range
  //  of 7 seconds using a 0.1 second time increment.
  if(rank==0)  
{
    for(i=0; i<600; ++i) {
      springRungeKutta4(&spring, dt);
 
      // springRK2(&spring, dt);
 
 
      // printf("time = %lf  x = %lf  vx = %lf\n",
      //         spring.s, spring.q[1], spring.q[0]);
    }
    
  }

// }
    MPI_Finalize();
    end = clock();
    //time count stops 
    total_time = ((double) (end - start)) / CLOCKS_PER_SEC;
    //calulate total time
    printf("\nExecution time: %f ", total_time);

  return 0;
}

//************************************************************
//  This method solves for the spring motion using a
//  4th-order Runge-Kutta solver
//************************************************************
void springRungeKutta4(struct SpringODE *spring, double ds) {

    int j;
    int numEqns;
    double s;
    double *q;
    double *dq1;
    double *dq2;
    double *dq3;
    double *dq4;

  //  Define a convenience variable to make the
  //  code more readable
    numEqns = spring->numEqns;

  //  Allocate memory for the arrays.
  q = (double *)malloc(numEqns*sizeof(double));
  dq1 = (double *)malloc(numEqns*sizeof(double));
  dq2 = (double *)malloc(numEqns*sizeof(double));
  dq3 = (double *)malloc(numEqns*sizeof(double));
  dq4 = (double *)malloc(numEqns*sizeof(double));

  //  Retrieve the current values of the dependent
  //  and independent variables.
  s = spring->s;

  if(rank==0)
  {
      // for(j=0; j<numEqns; ++j) {
      q[j] = spring->q[j];
  }
  if(rank==1)
  {
      // for(j=0; j<numEqns; ++j) {
      q[rank] = spring->q[rank];
  }

  // Compute the four Runge-Kutta steps, The return 
  // value of springRightHandSide method is an array of 
  // delta-q values for each of the four steps.
  springRightHandSide(spring, q, q,   ds, 0.0, dq1);
  springRightHandSide(spring, q, dq1, ds, 0.5, dq2);
  springRightHandSide(spring, q, dq2, ds, 0.5, dq3);
  springRightHandSide(spring, q, dq3, ds, 1.0, dq4);

  //  Update the dependent and independent variable values
  //  at the new dependent variable location and store the
  //  values in the ODE object arrays.
  spring->s = spring->s + ds;

  if(rank==0)
  { // for(j=0; j<numEqns; ++j) {  
    q[rank] = q[rank] + (dq1[rank] + 2.0*dq2[rank] + 2.0*dq3[rank] + dq4[rank])/6.0;
    spring->q[rank] = q[rank];
  }
  if(rank==1)     
  { // for(j=0; j<numEqns; ++j) {  
    q[rank] = q[rank] + (dq1[rank] + 2.0*dq2[rank] + 2.0*dq3[rank] + dq4[rank])/6.0;
    spring->q[rank] = q[rank];
  }

  //  Free up memory
  free(q);
  free(dq1);
  free(dq2);
  free(dq3);
  free(dq4);

  return;
}
void springRK2(struct SpringODE *spring, double ds) {

    int j;
    int numEqns;
    double s;
    double *q;
    double *dq1;
    double *dq2;

  //  Define a convenience variable to make the
  //  code more readable
    numEqns = spring->numEqns;

  //  Allocate memory for the arrays.
  q = (double *)malloc(numEqns*sizeof(double));
  dq1 = (double *)malloc(numEqns*sizeof(double));
  dq2 = (double *)malloc(numEqns*sizeof(double));

  //  Retrieve the current values of the dependent
  //  and independent variables.
  s = spring->s;

  if(rank==0)
  {
      // for(j=0; j<numEqns; ++j) {
      q[j] = spring->q[j];
  }
  if(rank==1)
  {
      // for(j=0; j<numEqns; ++j) {
      q[rank] = spring->q[rank];
  }

  // Compute the four Runge-Kutta steps, The return 
  // value of springRightHandSide method is an array of 
  // delta-q values for each of the four steps.
  springRightHandSide(spring, q, q,   ds, 0.0, dq1);
  springRightHandSide(spring, q, dq1, ds, 0.5, dq2);
  //  Update the dependent and independent variable values
  //  at the new dependent variable location and store the
  //  values in the ODE object arrays.
  spring->s = spring->s + ds;

  if(rank==0)
  { // for(j=0; j<numEqns; ++j) {  
    q[rank] = q[rank] + (0.5*dq1[rank] + 0.5*dq2[rank]);
    spring->q[rank] = q[rank];
  }
  if(rank==1)     
  { // for(j=0; j<numEqns; ++j) {  
    q[rank] = q[rank] + (0.5*dq1[rank] + 0.5*dq2[rank]);
    spring->q[rank] = q[rank];
  }

  //  Free up memory
  free(q);
  free(dq1);
  free(dq2);

  return;
}
 
//*************************************************************
//  This method loads the right-hand sides for the spring ODEs
//*************************************************************
void springRightHandSide( struct SpringODE *spring, double *q, 
             double *deltaQ, double ds, double qScale, double *dq) {
  //  q[0] = vx
  //  q[1] = x
  //  dq[0] = d(vx) = dt*(-mu*dxdt - k*x)/mass
  //  dq[1] = d(x) = dt*(v)
  double newQ[2]; // intermediate dependent variable values.
  double G = -9.81;
  double mass;
  double mu;
  double k;
  int i;
  // int rank;

  mass = spring->mass;
  mu = spring->mu;
  k = spring->k;

  //  Compute the intermediate values of the 
  //  dependent variables.
  if(rank==0)
{
//  for(i=0; i<2; ++i) {
    newQ[rank] = q[rank] + qScale*deltaQ[rank];
  
} 
  if(rank==1)
{
//  for(i=0; i<2; ++i) {
    newQ[rank] = q[rank] + qScale*deltaQ[rank];
  
} 
  //  Compute right-hand side values.
  dq[0] = ds*G - ds*(mu*newQ[0] + k*newQ[1])/mass;
  dq[1] = ds*(newQ[0]);

  return;
}