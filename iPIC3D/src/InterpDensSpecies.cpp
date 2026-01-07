#include "InterpDensSpecies.h"

/** allocated interpolated densities per species */
void interp_dens_species_allocate(struct grid* grd, struct interpDensSpecies* ids, int is)
{
    // set species ID
    ids->species_ID = is;

    // --- Node-based arrays: allocate as PINNED (copied back from GPU every cycle) ---
    ids->rhon = newArr3Pinned<FPinterp>(&ids->rhon_flat, grd->nxn, grd->nyn, grd->nzn);

    ids->Jx   = newArr3Pinned<FPinterp>(&ids->Jx_flat,   grd->nxn, grd->nyn, grd->nzn);
    ids->Jy   = newArr3Pinned<FPinterp>(&ids->Jy_flat,   grd->nxn, grd->nyn, grd->nzn);
    ids->Jz   = newArr3Pinned<FPinterp>(&ids->Jz_flat,   grd->nxn, grd->nyn, grd->nzn);

    ids->pxx  = newArr3Pinned<FPinterp>(&ids->pxx_flat,  grd->nxn, grd->nyn, grd->nzn);
    ids->pxy  = newArr3Pinned<FPinterp>(&ids->pxy_flat,  grd->nxn, grd->nyn, grd->nzn);
    ids->pxz  = newArr3Pinned<FPinterp>(&ids->pxz_flat,  grd->nxn, grd->nyn, grd->nzn);

    ids->pyy  = newArr3Pinned<FPinterp>(&ids->pyy_flat,  grd->nxn, grd->nyn, grd->nzn);
    ids->pyz  = newArr3Pinned<FPinterp>(&ids->pyz_flat,  grd->nxn, grd->nyn, grd->nzn);
    ids->pzz  = newArr3Pinned<FPinterp>(&ids->pzz_flat,  grd->nxn, grd->nyn, grd->nzn);

    // --- Center-based array: keep original allocator (typically CPU-only usage) ---
    ids->rhoc = newArr3<FPinterp>(&ids->rhoc_flat, grd->nxc, grd->nyc, grd->nzc);
}

/** deallocate interpolated densities per species */
void interp_dens_species_deallocate(struct grid* grd, struct interpDensSpecies* ids)
{
    // --- Free pinned node-based arrays ---
    delArr3Pinned(ids->rhon, ids->rhon_flat);
    ids->rhon = nullptr; ids->rhon_flat = nullptr;

    delArr3Pinned(ids->Jx, ids->Jx_flat);
    ids->Jx = nullptr; ids->Jx_flat = nullptr;

    delArr3Pinned(ids->Jy, ids->Jy_flat);
    ids->Jy = nullptr; ids->Jy_flat = nullptr;

    delArr3Pinned(ids->Jz, ids->Jz_flat);
    ids->Jz = nullptr; ids->Jz_flat = nullptr;

    delArr3Pinned(ids->pxx, ids->pxx_flat);
    ids->pxx = nullptr; ids->pxx_flat = nullptr;

    delArr3Pinned(ids->pxy, ids->pxy_flat);
    ids->pxy = nullptr; ids->pxy_flat = nullptr;

    delArr3Pinned(ids->pxz, ids->pxz_flat);
    ids->pxz = nullptr; ids->pxz_flat = nullptr;

    delArr3Pinned(ids->pyy, ids->pyy_flat);
    ids->pyy = nullptr; ids->pyy_flat = nullptr;

    delArr3Pinned(ids->pyz, ids->pyz_flat);
    ids->pyz = nullptr; ids->pyz_flat = nullptr;

    delArr3Pinned(ids->pzz, ids->pzz_flat);
    ids->pzz = nullptr; ids->pzz_flat = nullptr;

    // --- Free center-based array (non-pinned, allocated by newArr3) ---
    delArr3(ids->rhoc, grd->nxc, grd->nyc);
    ids->rhoc = nullptr; ids->rhoc_flat = nullptr;
}

/** deallocate interpolated densities per species */
void interpN2Crho(struct interpDensSpecies* ids, struct grid* grd){
    for (register int i = 1; i < grd->nxc - 1; i++)
        for (register int j = 1; j < grd->nyc - 1; j++)
            for (register int k = 1; k < grd->nzc - 1; k++){
                ids->rhoc[i][j][k] = .125 * (ids->rhon[i][j][k] + ids->rhon[i + 1][j][k] + ids->rhon[i][j + 1][k] + ids->rhon[i][j][k + 1] +
                                       ids->rhon[i + 1][j + 1][k]+ ids->rhon[i + 1][j][k + 1] + ids->rhon[i][j + 1][k + 1] + ids->rhon[i + 1][j + 1][k + 1]);
            }
}
