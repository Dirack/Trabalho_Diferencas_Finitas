/* Version 1.0 - Finite Diferences without borders condition

IMPORTANT: 'semborda.x' is a modified version of the 'sfTestfd2d' program from MADAGASCAR package

Purpose: FD modeling for a point source in a constant velocity model. The input is the velocity model, and the output are modeled filed snapshots and receiver records.

CHANGES: The program outpur is a shot gather instead of a receiver record.

Usage:

	<in.rsf semborda.x rec=receptor.rsf nb=30 nt=1000 dt=0.001 > out.rsf
	< receptor.rsf sfwigle > receptor.vpl
	sfpen receptor.vpl

Programmer: Rodolfo A. C. Neves (Dirack) 12/06/2019

Email:  rodolfo_profissional@hotmail.com  

Site: https://dirack.github.io

*/

#include <rsf.h>

#ifdef _OPENMP
#include <omp.h>
#endif

static int nb, nz, nx, nt, nzpad, nxpad;
static float dz, dx, dt, fm, c0, c11, c12, c21, c22;
static float *bndr, *wlt;
static float **vv, **p0, **p1, **ptr=NULL;

void expand2d(float** b, float** a)
/*< expand domain of 'a' to 'b': source(a)-->destination(b) >*/
{
    int iz,ix;

#ifdef _OPENMP
#pragma omp parallel for default(none)	\
	private(ix,iz)			\
	shared(b,a,nb,nz,nx)
#endif
    for     (ix=0;ix<nx;ix++) {
	for (iz=0;iz<nz;iz++) {
	    b[nb+ix][nb+iz] = a[ix][iz];
	}
    }

    for     (ix=0; ix<nxpad; ix++) {
	for (iz=0; iz<nb;    iz++) {
	    b[ix][      iz  ] = b[ix][nb  ];
	    b[ix][nzpad-iz-1] = b[ix][nzpad-nb-1];
	}
    }

    for     (ix=0; ix<nb;    ix++) {
	for (iz=0; iz<nzpad; iz++) {
	    b[ix 	 ][iz] = b[nb  		][iz];
	    b[nxpad-ix-1 ][iz] = b[nxpad-nb-1	][iz];
	}
    }
}


void window2d(float **a, float **b)
/*< window 'b' to 'a': source(b)-->destination(a) >*/
{
    int iz,ix;

#ifdef _OPENMP
#pragma omp parallel for default(none)	\
	private(ix,iz)			\
	shared(b,a,nb,nz,nx)
#endif
    for     (ix=0;ix<nx;ix++) {
	for (iz=0;iz<nz;iz++) {
	    a[ix][iz]=b[nb+ix][nb+iz] ;
	}
    }
}

void apply_sponge(float**p0, float **p1)
/*< apply absorbing boundary condition >*/
{
	int ix,iz;

#ifdef _OPENMP
#pragma omp parallel for	    \
    private(ix,iz)		    \
    shared(bndr,p0,p1)
#endif
	for(ix=0; ix<nxpad; ix++)
	{
		for(iz=0;iz<nb;iz++){	// top ABC			
			p0[ix][iz]=bndr[iz]*p0[ix][iz];
			p1[ix][iz]=bndr[iz]*p1[ix][iz];
		}
		for(iz=nz+nb;iz<nzpad;iz++){// bottom ABC			
			p0[ix][iz]=bndr[nzpad-iz-1]*p0[ix][iz];
			p1[ix][iz]=bndr[nzpad-iz-1]*p1[ix][iz];
		}
	}

#ifdef _OPENMP
#pragma omp parallel for	    \
    private(ix,iz)		    \
    shared(bndr,p0,p1)
#endif
	for(iz=0; iz<nzpad; iz++)
	{
		for(ix=0;ix<nb;ix++){	// left ABC			
			p0[ix][iz]=bndr[ix]*p0[ix][iz];
			p1[ix][iz]=bndr[ix]*p1[ix][iz];
		}	
		for(ix=nx+nb;ix<nxpad;ix++){// right ABC			
			p0[ix][iz]=bndr[nxpad-ix-1]*p0[ix][iz];
			p1[ix][iz]=bndr[nxpad-ix-1]*p1[ix][iz];
		}	
	}
}


void step_forward(float **p0, float **p1)
/*< forward modeling step >*/
{
	int ix,iz;
	float tmp;

#ifdef _OPENMP
#pragma omp parallel for	    \
    private(ix,iz,tmp)		    \
    shared(p0,p1,vv,c0,c11,c12,c21,c22,nzpad,nxpad)
#endif	
	for (ix=2; ix < nxpad-2; ix++) 
	for (iz=2; iz < nzpad-2; iz++) 
	{
		tmp =	c0*p1[ix][iz]+
			c11*(p1[ix][iz-1]+p1[ix][iz+1])+
			c12*(p1[ix][iz-2]+p1[ix][iz+2])+
			c21*(p1[ix-1][iz]+p1[ix+1][iz])+
			c22*(p1[ix-2][iz]+p1[ix+2][iz]);
		p0[ix][iz]=2*p1[ix][iz]-p0[ix][iz]+vv[ix][iz]*tmp;
	}
}


int main(int argc, char* argv[])
{
	int jt, ft, it, ib, ix, iz, sx, sz;
	int nr,ir; /* Posição do receptor */
	float tmp;
	float **v0;
	float **r1; /* receptor */
	bool border; /* Aplicar condição de borda */
	sf_file Fv, Fw, r;
	sf_axis eixot, eixox; /* eixo do tempo */

    	sf_init(argc,argv);
#ifdef _OPENMP
    	omp_init();
#endif

	Fv = sf_input("in");/* Modelo de velocidades */
	Fw = sf_output("out");/* Snapshots do campo de ondas */
	r = sf_output("rec"); /* Shot gather */

    	if (!sf_histint(Fv,"n1",&nz)) sf_error("No n1= in input");/* Modelo de velocidades: nz */
    	if (!sf_histint(Fv,"n2",&nx)) sf_error("No n2= in input");/* Modelo de velocidades: nx */
    	if (!sf_histfloat(Fv,"d1",&dz)) sf_error("No d1= in input");/* Modelo de velocidades: dz */
    	if (!sf_histfloat(Fv,"d2",&dx)) sf_error("No d2= in input");/* Modelo de velocidades: dx */
    	if (!sf_getint("nb",&nb)) nb=30; /* Comprimento da borda absorvente ABC */
    	if (!sf_getint("nt",&nt)) sf_error("nt required");/* Número de passos no tempo */
    	if (!sf_getfloat("dt",&dt)) sf_error("dt required");/* Intervalo de amostragem */
    	if (!sf_getfloat("fm",&fm)) fm=20.0; /* Frequência dominante do pulso Ricker */
   	if (!sf_getint("ft",&ft)) ft=0; /* Primeiro tempo de registro */
    	if (!sf_getint("jt",&jt)) jt=1;	/* Intervalo de tempo */
	if (!sf_getbool("border",&border)) border=0;

	sf_putint(Fw,"n1",nz);
	sf_putint(Fw,"n2",nx);
    	sf_putint(Fw,"n3",(nt-ft)/jt);
    	sf_putfloat(Fw,"d3",jt*dt);
    	sf_putfloat(Fw,"o3",ft*dt);

	nzpad=nz+2*nb;
	nxpad=nx+2*nb;

	// Posição da fonte
	sx=nxpad/2;
	sz=nzpad/2;

	if (!sf_getint("nr",&nr)) nr=nxpad/2;	/* Posição do receptor */

	/*< initialize 4-th order fd coefficients >*/
	tmp = 1.0/(dz*dz);
	c11 = 4.0*tmp/3.0;
	c12= -tmp/12.0;
	tmp = 1.0/(dx*dx);
	c21 = 4.0*tmp/3.0;
	c22= -tmp/12.0;
	c0=-2.0*(c11+c12+c21+c22);

	wlt=sf_floatalloc(nt);
	bndr=sf_floatalloc(nb);
	v0=sf_floatalloc2(nz, nx); 	
	vv=sf_floatalloc2(nzpad, nxpad);
	p0=sf_floatalloc2(nzpad, nxpad);
	p1=sf_floatalloc2(nzpad, nxpad);
	
	/* Traço sísmico do receptor r1 */
	r1=sf_floatalloc2(nt,nx); 
	eixot = sf_maxa(nt,0,dt);
	eixox = sf_maxa(nx,0,1);
	sf_setlabel(eixot,"Tempo");
	sf_setlabel(eixox,"Receptor");
	sf_oaxa(r,eixot,1);
	sf_oaxa(r,eixox,2);
	sf_putstring(r,"unit1","s");
	sf_putstring(r,"unit2","x");

	for(it=0;it<nt;it++){
		tmp=SF_PI*fm*(it*dt-1.0/fm);tmp*=tmp;
		wlt[it]=(1.0-2.0*tmp)*expf(-tmp);
	}
	for(ib=0;ib<nb;ib++){
		tmp=0.015*(nb-ib);
		bndr[ib]=expf(-tmp*tmp);
	}
	sf_floatread(v0[0],nz*nx,Fv);
	expand2d(vv, v0);
	for(ix=0;ix<nxpad;ix++){
	    for(iz=0;iz<nzpad;iz++){
		tmp=vv[ix][iz]*dt;
		vv[ix][iz]=tmp*tmp;
	    }
	}
	memset(p0[0],0,nzpad*nxpad*sizeof(float));
	memset(p1[0],0,nzpad*nxpad*sizeof(float));

	for(it=0; it<nt; it++)
	{
		if(it>=ft)
		{
			window2d(v0, p0);
			sf_floatwrite(v0[0], nz*nx, Fw);

			for(ir=0;ir<nx;ir++){

				/* Registro no receptor r1 */
				r1[ir][it]=v0[ir][35];

			}
			
		}

		p1[nxpad/2][35]+=wlt[it]; // Onda direta

		p1[sx][sz]+=wlt[it]; //Injeção da fonte
		step_forward(p0, p1);

		/* Função para aplicar a borda */
		if(border) apply_sponge(p0, p1);

		ptr=p0; p0=p1; p1=ptr;

	}

	/* Escreva o traço sísmico do receptor r1 */
	sf_floatwrite(r1[0],nx*nt,r);

	free(wlt);
	free(*v0); free(v0);
	free(*vv); free(vv);
	free(*p0); free(p0);
	free(*p1); free(p1);
	free(bndr);
    	exit(0);
}

