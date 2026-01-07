#pragma once
#include "Parameters.h"
#include "Grid.h"
#include "Particles.h"
#include "InterpDensSpecies.h"
#include "EMfield.h"

// ------------------------------
// GPU mover 
// ------------------------------

/** Initialize and allocate GPU resources for the particle mover (one-time). */
void mover_gpu_init(struct parameters* param, struct grid* grd, struct EMfield* field, struct particles* parts);

/** Copy updated electromagnetic field arrays (E, B) from host to device (once per cycle). */
void mover_gpu_update_fields(struct grid* grd, struct EMfield* field);

/** Run the particle mover on the GPU for one species and copy updated particles back to host. */
int mover_PC_GPU(struct particles* part, struct grid* grd, struct parameters* param);

/** Release GPU resources associated with the particle mover. */
void mover_gpu_finalize();

// ------------------------------
// GPU interpolation (P2G) 
// ------------------------------

/** Initialize and allocate GPU resources for P2G interpolation (one-time). */
void interp_gpu_init(struct parameters* param, struct grid* grd, struct interpDensSpecies* ids_host);

/** Run P2G interpolation (particle -> grid) on the GPU for one species and copy densities back to host. */
void interpP2G_GPU(struct particles* part, struct interpDensSpecies* ids_host, struct grid* grd);

/** Release GPU resources associated with P2G interpolation. */
void interp_gpu_finalize();
