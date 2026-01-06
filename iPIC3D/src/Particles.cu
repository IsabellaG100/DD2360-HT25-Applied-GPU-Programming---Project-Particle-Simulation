#include "Particles.h"
#include "Alloc.h"
#include "PrecisionTypes.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

//** CUDA error checking helper*/
static inline void cudaCheck(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::abort();
    }
}

//** Device-side storage for one species*/
struct DeviceParticles {
    FPpart* x = nullptr;
    FPpart* y = nullptr;
    FPpart* z = nullptr;
    FPpart* u = nullptr;
    FPpart* v = nullptr;
    FPpart* w = nullptr;

    long capacity_npmax = 0;  // allocated length
};

//** Global GPU context for mover (allocated once, reused each cycle) */
struct MoverGPUContext {
    bool initialized = false;

    int  ns = 0;

    // node dimensions for flattening node-based arrays
    int nxn = 0;
    int nyn = 0;
    int nzn = 0;
    long Nnodes = 0;

    // cached grid scalars used by mover
    FPfield invdx = 0.0f, invdy = 0.0f, invdz = 0.0f, invVOL = 0.0f;
    FPpart  xStart = 0.0f, yStart = 0.0f, zStart = 0.0f;
    FPpart  Lx = 0.0f, Ly = 0.0f, Lz = 0.0f;

    // cached simulation constant(s)
    FPpart c = 0.0f;

    // device grid node coordinates (flat)
    FPfield* d_XN = nullptr;
    FPfield* d_YN = nullptr;
    FPfield* d_ZN = nullptr;

    // device fields (flat, node-based)
    FPfield* d_Ex = nullptr;
    FPfield* d_Ey = nullptr;
    FPfield* d_Ez = nullptr;
    FPfield* d_Bx = nullptr;
    FPfield* d_By = nullptr;
    FPfield* d_Bz = nullptr;

    // per species device particle buffers
    DeviceParticles* d_parts = nullptr;
};

static MoverGPUContext g_ctx;

// ------------------------------------------------------------
// CPU Mover
// ------------------------------------------------------------

/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
                                                                        
    return(0); // exit succcesfully
} // end of the mover

// ------------------------------------------------------------
// GPU Mover
// ------------------------------------------------------------

//** CUDA kernel implementing the particle mover: one thread updates one particle
//** over all subcycles and iterations */
__global__ void mover_PC_kernel(
    // particles (device)
    FPpart* x, FPpart* y, FPpart* z,
    FPpart* u, FPpart* v, FPpart* w,
    long nop,

    // fields on nodes (device, flat)
    const FPfield* Ex, const FPfield* Ey, const FPfield* Ez,
    const FPfield* Bx, const FPfield* By, const FPfield* Bz,

    // grid node coordinates (device, flat)
    const FPfield* XN, const FPfield* YN, const FPfield* ZN,

    // node strides for flattening
    int nyn, int nzn,

    // grid scalars
    FPfield invdx, FPfield invdy, FPfield invdz, FPfield invVOL,
    FPpart xStart, FPpart yStart, FPpart zStart,
    FPpart Lx, FPpart Ly, FPpart Lz,
    int periodicX, int periodicY, int periodicZ,

    // mover scalars
    FPpart dt,
    int n_sub_cycles,
    int NiterMover,
    FPpart qom,
    FPpart c,

    // node dimensions (for defensive clamping)
    int nxn, int nyn_dim, int nzn_dim
)
{
    long i = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    if (i >= nop) return;

    // Load particle into registers
    FPpart px = x[i], py = y[i], pz = z[i];
    FPpart pu = u[i], pv = v[i], pw = w[i];

    // Time step split into sub-cycles (same as CPU)
    FPpart dt_sub = dt / (FPpart)n_sub_cycles;
    FPpart dto2   = (FPpart)0.5f * dt_sub;
    FPpart qomdt2 = qom * dto2 / c;

    for (int sub = 0; sub < n_sub_cycles; sub++) {

        // Save start-of-subcycle position
        FPpart xpt = px, ypt = py, zpt = pz;

        FPpart upt = 0.0f, vpt = 0.0f, wpt = 0.0f;

        // Iterative average-velocity solve (same structure as CPU)
        for (int it = 0; it < NiterMover; it++) {

            int ix = 2 + (int)((px - xStart) * invdx);
            int iy = 2 + (int)((py - yStart) * invdy);
            int iz = 2 + (int)((pz - zStart) * invdz);

            // Defensive clamping to avoid illegal indexing if a particle goes out-of-range
            // We need ix-1 and ix to be valid, same for iy/iz, and we also access ix-ii with ii in {0,1}.
            // So ix must be in [1, nxn-2], similarly for y,z.
            if (ix < 1) ix = 1;
            if (iy < 1) iy = 1;
            if (iz < 1) iz = 1;
            if (ix > nxn - 2) ix = nxn - 2;
            if (iy > nyn_dim - 2) iy = nyn_dim - 2;
            if (iz > nzn_dim - 2) iz = nzn_dim - 2;

            // Distances to surrounding nodes (same as CPU, but flat indexing)
            // Alloc.h flatten rule: get_idx(x,y,z, stride_y, stride_z)
            FPfield xi0   = (FPfield)(px - XN[get_idx(ix - 1, iy,     iz,     (long)nyn, (long)nzn)]);
            FPfield eta0  = (FPfield)(py - YN[get_idx(ix,     iy - 1, iz,     (long)nyn, (long)nzn)]);
            FPfield zeta0 = (FPfield)(pz - ZN[get_idx(ix,     iy,     iz - 1, (long)nyn, (long)nzn)]);

            FPfield xi1   = (FPfield)(XN[get_idx(ix, iy, iz, (long)nyn, (long)nzn)] - px);
            FPfield eta1  = (FPfield)(YN[get_idx(ix, iy, iz, (long)nyn, (long)nzn)] - py);
            FPfield zeta1 = (FPfield)(ZN[get_idx(ix, iy, iz, (long)nyn, (long)nzn)] - pz);

            // Interpolate E and B from 8 surrounding nodes (2x2x2)
            FPfield Exl = 0.0f, Eyl = 0.0f, Ezl = 0.0f;
            FPfield Bxl = 0.0f, Byl = 0.0f, Bzl = 0.0f;

            for (int ii = 0; ii < 2; ii++) {
                FPfield wx = (ii == 0) ? xi0 : xi1;
                for (int jj = 0; jj < 2; jj++) {
                    FPfield wy = (jj == 0) ? eta0 : eta1;
                    for (int kk = 0; kk < 2; kk++) {
                        FPfield wz = (kk == 0) ? zeta0 : zeta1;

                        FPfield wgt = wx * wy * wz * invVOL;

                        long gi = get_idx(ix - ii, iy - jj, iz - kk, (long)nyn, (long)nzn);

                        Exl += wgt * Ex[gi];
                        Eyl += wgt * Ey[gi];
                        Ezl += wgt * Ez[gi];

                        Bxl += wgt * Bx[gi];
                        Byl += wgt * By[gi];
                        Bzl += wgt * Bz[gi];
                    }
                }
            }

            // Boris-like update (same equations as CPU mover_PC)
            FPpart omdtsq = qomdt2 * qomdt2 *
                ((FPpart)Bxl * (FPpart)Bxl + (FPpart)Byl * (FPpart)Byl + (FPpart)Bzl * (FPpart)Bzl);

            FPpart denom = (FPpart)1.0f / ((FPpart)1.0f + omdtsq);

            FPpart ut = pu + qomdt2 * (FPpart)Exl;
            FPpart vt = pv + qomdt2 * (FPpart)Eyl;
            FPpart wt = pw + qomdt2 * (FPpart)Ezl;

            FPpart udotb = ut * (FPpart)Bxl + vt * (FPpart)Byl + wt * (FPpart)Bzl;

            upt = (ut + qomdt2 * (vt * (FPpart)Bzl - wt * (FPpart)Byl + qomdt2 * udotb * (FPpart)Bxl)) * denom;
            vpt = (vt + qomdt2 * (wt * (FPpart)Bxl - ut * (FPpart)Bzl + qomdt2 * udotb * (FPpart)Byl)) * denom;
            wpt = (wt + qomdt2 * (ut * (FPpart)Byl - vt * (FPpart)Bxl + qomdt2 * udotb * (FPpart)Bzl)) * denom;

            // Half-step position update inside the iteration loop (same as CPU)
            px = xpt + upt * dto2;
            py = ypt + vpt * dto2;
            pz = zpt + wpt * dto2;
        }

        // Final update for this subcycle (same as CPU)
        pu = (FPpart)2.0f * upt - pu;
        pv = (FPpart)2.0f * vpt - pv;
        pw = (FPpart)2.0f * wpt - pw;

        px = xpt + upt * dt_sub;
        py = ypt + vpt * dt_sub;
        pz = zpt + wpt * dt_sub;

        // Boundary conditions (same logic as CPU mover_PC)
        // X direction
        if (px > Lx) {
            if (periodicX) px -= Lx;
            else { pu = -pu; px = (FPpart)2.0f * Lx - px; }
        }
        if (px < (FPpart)0.0f) {
            if (periodicX) px += Lx;
            else { pu = -pu; px = -px; }
        }

        // Y direction
        if (py > Ly) {
            if (periodicY) py -= Ly;
            else { pv = -pv; py = (FPpart)2.0f * Ly - py; }
        }
        if (py < (FPpart)0.0f) {
            if (periodicY) py += Ly;
            else { pv = -pv; py = -py; }
        }

        // Z direction
        if (pz > Lz) {
            if (periodicZ) pz -= Lz;
            else { pw = -pw; pz = (FPpart)2.0f * Lz - pz; }
        }
        if (pz < (FPpart)0.0f) {
            if (periodicZ) pz += Lz;
            else { pw = -pw; pz = -pz; }
        }
    }

    // Store back to global memory
    x[i] = px; y[i] = py; z[i] = pz;
    u[i] = pu; v[i] = pv; w[i] = pw;
}

//** One-time initialization of GPU resources for the particle mover */
void mover_gpu_init(struct parameters* param, struct grid* grd, struct EMfield* field, struct particles* parts)
{
    if (g_ctx.initialized) return;

    g_ctx.initialized = true;
    g_ctx.ns  = param->ns;

    g_ctx.nxn = grd->nxn;
    g_ctx.nyn = grd->nyn;
    g_ctx.nzn = grd->nzn;
    g_ctx.Nnodes = (long)g_ctx.nxn * (long)g_ctx.nyn * (long)g_ctx.nzn;

    g_ctx.invdx  = grd->invdx;
    g_ctx.invdy  = grd->invdy;
    g_ctx.invdz  = grd->invdz;
    g_ctx.invVOL = grd->invVOL;

    g_ctx.xStart = (FPpart)grd->xStart;
    g_ctx.yStart = (FPpart)grd->yStart;
    g_ctx.zStart = (FPpart)grd->zStart;

    g_ctx.Lx = (FPpart)grd->Lx;
    g_ctx.Ly = (FPpart)grd->Ly;
    g_ctx.Lz = (FPpart)grd->Lz;

    g_ctx.c  = (FPpart)param->c;

    // Allocate and copy grid node coordinates ONCE
    cudaCheck(cudaMalloc((void**)&g_ctx.d_XN, g_ctx.Nnodes * sizeof(FPfield)), "cudaMalloc d_XN");
    cudaCheck(cudaMalloc((void**)&g_ctx.d_YN, g_ctx.Nnodes * sizeof(FPfield)), "cudaMalloc d_YN");
    cudaCheck(cudaMalloc((void**)&g_ctx.d_ZN, g_ctx.Nnodes * sizeof(FPfield)), "cudaMalloc d_ZN");

    cudaCheck(cudaMemcpy(g_ctx.d_XN, grd->XN_flat, g_ctx.Nnodes * sizeof(FPfield), cudaMemcpyHostToDevice), "Memcpy XN_flat");
    cudaCheck(cudaMemcpy(g_ctx.d_YN, grd->YN_flat, g_ctx.Nnodes * sizeof(FPfield), cudaMemcpyHostToDevice), "Memcpy YN_flat");
    cudaCheck(cudaMemcpy(g_ctx.d_ZN, grd->ZN_flat, g_ctx.Nnodes * sizeof(FPfield), cudaMemcpyHostToDevice), "Memcpy ZN_flat");

    // Allocate field arrays on device (contents updated each cycle)
    cudaCheck(cudaMalloc((void**)&g_ctx.d_Ex, g_ctx.Nnodes * sizeof(FPfield)), "cudaMalloc d_Ex");
    cudaCheck(cudaMalloc((void**)&g_ctx.d_Ey, g_ctx.Nnodes * sizeof(FPfield)), "cudaMalloc d_Ey");
    cudaCheck(cudaMalloc((void**)&g_ctx.d_Ez, g_ctx.Nnodes * sizeof(FPfield)), "cudaMalloc d_Ez");
    cudaCheck(cudaMalloc((void**)&g_ctx.d_Bx, g_ctx.Nnodes * sizeof(FPfield)), "cudaMalloc d_Bx");
    cudaCheck(cudaMalloc((void**)&g_ctx.d_By, g_ctx.Nnodes * sizeof(FPfield)), "cudaMalloc d_By");
    cudaCheck(cudaMalloc((void**)&g_ctx.d_Bz, g_ctx.Nnodes * sizeof(FPfield)), "cudaMalloc d_Bz");

    // Allocate per-species particle arrays on device and copy initial data
    g_ctx.d_parts = new DeviceParticles[g_ctx.ns];

    for (int is = 0; is < g_ctx.ns; is++) {
        DeviceParticles& dp = g_ctx.d_parts[is];
        dp.capacity_npmax = parts[is].npmax;

        cudaCheck(cudaMalloc((void**)&dp.x, dp.capacity_npmax * sizeof(FPpart)), "cudaMalloc dp.x");
        cudaCheck(cudaMalloc((void**)&dp.y, dp.capacity_npmax * sizeof(FPpart)), "cudaMalloc dp.y");
        cudaCheck(cudaMalloc((void**)&dp.z, dp.capacity_npmax * sizeof(FPpart)), "cudaMalloc dp.z");
        cudaCheck(cudaMalloc((void**)&dp.u, dp.capacity_npmax * sizeof(FPpart)), "cudaMalloc dp.u");
        cudaCheck(cudaMalloc((void**)&dp.v, dp.capacity_npmax * sizeof(FPpart)), "cudaMalloc dp.v");
        cudaCheck(cudaMalloc((void**)&dp.w, dp.capacity_npmax * sizeof(FPpart)), "cudaMalloc dp.w");

        // Copy only the active particles (nop)
        long nop = parts[is].nop;
        cudaCheck(cudaMemcpy(dp.x, parts[is].x, nop * sizeof(FPpart), cudaMemcpyHostToDevice), "Memcpy part.x H2D");
        cudaCheck(cudaMemcpy(dp.y, parts[is].y, nop * sizeof(FPpart), cudaMemcpyHostToDevice), "Memcpy part.y H2D");
        cudaCheck(cudaMemcpy(dp.z, parts[is].z, nop * sizeof(FPpart), cudaMemcpyHostToDevice), "Memcpy part.z H2D");
        cudaCheck(cudaMemcpy(dp.u, parts[is].u, nop * sizeof(FPpart), cudaMemcpyHostToDevice), "Memcpy part.u H2D");
        cudaCheck(cudaMemcpy(dp.v, parts[is].v, nop * sizeof(FPpart), cudaMemcpyHostToDevice), "Memcpy part.v H2D");
        cudaCheck(cudaMemcpy(dp.w, parts[is].w, nop * sizeof(FPpart), cudaMemcpyHostToDevice), "Memcpy part.w H2D");
    }

    // Copy initial fields once
    mover_gpu_update_fields(grd, field);
}

//** Update fields once per cycle (host -> device) */
void mover_gpu_update_fields(struct grid* /*grd*/, struct EMfield* field)
{
    if (!g_ctx.initialized) {
        std::fprintf(stderr, "mover_gpu_update_fields called before mover_gpu_init\n");
        std::abort();
    }

    cudaCheck(cudaMemcpy(g_ctx.d_Ex, field->Ex_flat, g_ctx.Nnodes * sizeof(FPfield), cudaMemcpyHostToDevice), "Memcpy Ex_flat");
    cudaCheck(cudaMemcpy(g_ctx.d_Ey, field->Ey_flat, g_ctx.Nnodes * sizeof(FPfield), cudaMemcpyHostToDevice), "Memcpy Ey_flat");
    cudaCheck(cudaMemcpy(g_ctx.d_Ez, field->Ez_flat, g_ctx.Nnodes * sizeof(FPfield), cudaMemcpyHostToDevice), "Memcpy Ez_flat");

    cudaCheck(cudaMemcpy(g_ctx.d_Bx, field->Bxn_flat, g_ctx.Nnodes * sizeof(FPfield), cudaMemcpyHostToDevice), "Memcpy Bxn_flat");
    cudaCheck(cudaMemcpy(g_ctx.d_By, field->Byn_flat, g_ctx.Nnodes * sizeof(FPfield), cudaMemcpyHostToDevice), "Memcpy Byn_flat");
    cudaCheck(cudaMemcpy(g_ctx.d_Bz, field->Bzn_flat, g_ctx.Nnodes * sizeof(FPfield), cudaMemcpyHostToDevice), "Memcpy Bzn_flat");
}

//** Move one species on GPU, then copy particles back to host */
int mover_PC_GPU(struct particles* part, struct grid* /*grd*/, struct parameters* param)
{
    if (!g_ctx.initialized) {
        std::fprintf(stderr, "mover_PC_GPU called before mover_gpu_init\n");
        return -1;
    }

    const int is = part->species_ID;
    if (is < 0 || is >= g_ctx.ns) {
        std::fprintf(stderr, "mover_PC_GPU: invalid species_ID = %d\n", is);
        return -1;
    }

    DeviceParticles& dp = g_ctx.d_parts[is];

    const long nop = part->nop;

    // Launch configuration: one thread per particle
    const int threads = 256;
    const int blocks  = (int)((nop + threads - 1) / threads);

    const int periodicX = param->PERIODICX ? 1 : 0;
    const int periodicY = param->PERIODICY ? 1 : 0;
    const int periodicZ = param->PERIODICZ ? 1 : 0;

    mover_PC_kernel<<<blocks, threads>>>(
        dp.x, dp.y, dp.z,
        dp.u, dp.v, dp.w,
        nop,
        g_ctx.d_Ex, g_ctx.d_Ey, g_ctx.d_Ez,
        g_ctx.d_Bx, g_ctx.d_By, g_ctx.d_Bz,
        g_ctx.d_XN, g_ctx.d_YN, g_ctx.d_ZN,
        g_ctx.nyn, g_ctx.nzn,
        g_ctx.invdx, g_ctx.invdy, g_ctx.invdz, g_ctx.invVOL,
        g_ctx.xStart, g_ctx.yStart, g_ctx.zStart,
        g_ctx.Lx, g_ctx.Ly, g_ctx.Lz,
        periodicX, periodicY, periodicZ,
        (FPpart)param->dt,
        part->n_sub_cycles,
        part->NiterMover,
        part->qom,
        g_ctx.c,
        g_ctx.nxn, g_ctx.nyn, g_ctx.nzn
    );

    cudaCheck(cudaGetLastError(), "kernel launch mover_PC_kernel");
    cudaCheck(cudaDeviceSynchronize(), "kernel sync mover_PC_kernel");

    // Copy results back to host so CPU pipeline (interp/output) works unchanged
    cudaCheck(cudaMemcpy(part->x, dp.x, nop * sizeof(FPpart), cudaMemcpyDeviceToHost), "Memcpy x D2H");
    cudaCheck(cudaMemcpy(part->y, dp.y, nop * sizeof(FPpart), cudaMemcpyDeviceToHost), "Memcpy y D2H");
    cudaCheck(cudaMemcpy(part->z, dp.z, nop * sizeof(FPpart), cudaMemcpyDeviceToHost), "Memcpy z D2H");
    cudaCheck(cudaMemcpy(part->u, dp.u, nop * sizeof(FPpart), cudaMemcpyDeviceToHost), "Memcpy u D2H");
    cudaCheck(cudaMemcpy(part->v, dp.v, nop * sizeof(FPpart), cudaMemcpyDeviceToHost), "Memcpy v D2H");
    cudaCheck(cudaMemcpy(part->w, dp.w, nop * sizeof(FPpart), cudaMemcpyDeviceToHost), "Memcpy w D2H");

    return 0;
}

//** Free GPU resources at end of program*/
void mover_gpu_finalize()
{
    if (!g_ctx.initialized) return;

    // per-species particle buffers
    if (g_ctx.d_parts) {
        for (int is = 0; is < g_ctx.ns; is++) {
            DeviceParticles& dp = g_ctx.d_parts[is];
            if (dp.x) cudaFree(dp.x);
            if (dp.y) cudaFree(dp.y);
            if (dp.z) cudaFree(dp.z);
            if (dp.u) cudaFree(dp.u);
            if (dp.v) cudaFree(dp.v);
            if (dp.w) cudaFree(dp.w);
        }
        delete[] g_ctx.d_parts;
        g_ctx.d_parts = nullptr;
    }

    // grid node coordinates
    if (g_ctx.d_XN) cudaFree(g_ctx.d_XN);
    if (g_ctx.d_YN) cudaFree(g_ctx.d_YN);
    if (g_ctx.d_ZN) cudaFree(g_ctx.d_ZN);

    // fields
    if (g_ctx.d_Ex) cudaFree(g_ctx.d_Ex);
    if (g_ctx.d_Ey) cudaFree(g_ctx.d_Ey);
    if (g_ctx.d_Ez) cudaFree(g_ctx.d_Ez);
    if (g_ctx.d_Bx) cudaFree(g_ctx.d_Bx);
    if (g_ctx.d_By) cudaFree(g_ctx.d_By);
    if (g_ctx.d_Bz) cudaFree(g_ctx.d_Bz);

    g_ctx = MoverGPUContext{};
} // End of GPU mover

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
