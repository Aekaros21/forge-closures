# Data

The inputs of the cases in `reproduce/cases.json`: where each comes from, its terms of use, and how to obtain the
files that cannot be redistributed here. `data/SHA256SUMS` holds the checksums of the files used in the paper, and

```sh
python3 data/fetch_data.py check
```

verifies every input that is present and reports each source separately, with the command or page that supplies
what is missing.

## Included in this repository

These files are works of NASA, in the public domain in the United States, and are included unchanged.

| Folder | Content | Source |
|---|---|---|
| `data/assets/grids/plate` | Flat-plate grids, five nested levels (35×25 to 545×385); the cases use 137×97 | NASA TMR, 2-D zero-pressure-gradient flat plate |
| `data/assets/grids/bump` | Bump-in-channel grids 89×41, 177×81 and 353×161; the cases use 177×81 | NASA TMR, 2-D bump in channel |
| `data/assets/grids/naca` | NACA 0012 grids 113×33, 225×65 and 449×129; the cases use 225×65 | NASA TMR, 2-D NACA 0012 airfoil |
| `data/assets/grids/hump` | Wall-mounted hump grids 103×28, 205×55 and 409×109; the cases use 409×109 | NASA TMR, 2-D wall-mounted hump |
| `data/assets/references/tmr` | CFL3D SST solutions of the plate and bump; CFL3D solutions and NASA experimental data for the NACA 0012 | NASA TMR; Ladson (1988) |
| `data/assets/references/hump` | Hump experiment without flow control: skin friction, surface pressure, inflow profile and PIV velocity and Reynolds stress | Greenblatt et al. (2006), Naughton et al. (2006), via NASA TMR |
| `cases/faith` | FAITH hill: centerline PIV mean velocity (the 2 Hz set, streamwise and vertical components), fringe-imaging skin friction, model geometry and the original notes | Bell et al. (2012), via NASA TMR |

The NASA Turbulence Modeling Resource (TMR, https://turbmodels.larc.nasa.gov/) now redirects to
https://www.nasa.gov/nasa-turbulence-modeling-resource/, which serves only file archives, so the case pages and
reference tables were retrieved from the Internet Archive captures of the original addresses. The coarser and
finer grid levels are included for grid studies such as the one in Appendix B of the paper.

The plate and bump are compared with the CFL3D solutions of SST rather than with experiments. OpenFOAM's
`kOmegaSST` uses the strain-based production and eddy-viscosity limiter of the 2003 form of SST, so the bump is
compared with the SST-2003 solution (`Bump_SST2003/`); the 1994 form (`Bump_SST/`) is kept for comparison. On the
flat plate the two forms coincide. The NACA 0012 cases are compared with Ladson's tripped force measurements
(`NACA0012_validation_CLCD_Ladson_expdata.dat`), the set NASA recommends for fully turbulent simulations.

## Downloaded by `data/fetch_data.py`

| Command | Content | Source | Size |
|---|---|---|---|
| `pch22` | Rotating-channel DNS at seven rotation numbers, Ro = 0 to 0.50 (`f2_r_00.dat` to `f2_r_50.dat`) | Kristoffersen and Andersson (1993), AGARD database case PCH22, from the mirror at https://torroja.dmt.upm.es/turbdata/agard/chapter5/PCH22/f2/ | 25 kB |
| `ducts` | Square-duct DNS fields (Pinelli et al. 2010) and SST template cases at Re_b = 2000 and 3200 | McConkey et al. (2021) dataset on Kaggle, `ryleymcconkey/ml-turbulence-dataset` | 380 MB |

The mirror of the AGARD database serves an incomplete TLS certificate chain, so `pch22` does not verify the
certificate; each file is checked against its SHA-256 checksum instead. `ducts` uses the Kaggle command-line client
(`pip install kaggle`, with an API token from the Kaggle account settings saved as `~/.kaggle/kaggle.json`) and
downloads only the files the two duct cases need. It then rebuilds the scorer's reference arrays in
`data/mcconkey/cache/` from the DNS fields, as for the paper.

## ERCOFTAC data of the three-dimensional cases

The data of the Ahmed bodies, the Stanford diffuser and the wing–body junction are distributed by ERCOFTAC under
its own terms and are not included. Download them from the pages below, place the files in the listed folders
with their names unchanged, and run `python3 data/fetch_data.py check`.

| Folder | Files | Source |
|---|---|---|
| `data/assets/references/ahmed/` | `ahmed-front-geo.dat`; for each slant `S` = 25 and 35: `ahmed-S-press.dat`, `ahmed-S-yp000-xz.dat`, `ahmed-S-yp100-xz.dat`, `ahmed-S-yp180-xz.dat`, `ahmed-S-xp000-yz.dat`, `ahmed-S-xp080-yz.dat`, `ahmed-S-xp200-yz.dat`, `ahmed-S-xp500-yz.dat` | Lienhart and Becker (2003), ERCOFTAC Classic Collection case 082, http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case082 |
| `data/assets/references/diffuser3d/` | `Diffuser 1 data.mat` (from `Diffuser_1_data.zip`), `UFR4-16_Cp_Re%3D10000.xlsx` (the name the download link gives; rename it if a browser saves it as `UFR4-16_Cp_Re=10000.xlsx`) | Cherry, Elkins and Eaton (2008), ERCOFTAC Knowledge Base UFR 4-16, https://kbwiki.ercoftac.org/w/index.php/UFR_4-16 |
| `data/assets/references/wingbody/` | `wbj-etab1.dat`, `wbj-cp-body-27.dat`, `wbj-cp-wing-27.dat`, `wbj-el01tc.dat` to `wbj-el11tc.dat`, `wbj-fp05tc.dat`, `wbj-fp06tc.dat`, `wbj-fp07tc.dat`, `wbj-fp09tc.dat`, `wbj-fp10tc.dat`, `wbj-fp11tc.dat`, `wbj-lp05tc.dat`, `wbj-lp08tc.dat`, `wbj-lp10tc.dat` | Devenport and Simpson (1990), Fleming et al. (1993), ERCOFTAC Classic Collection case 008, http://cfd.mace.manchester.ac.uk/ercoftac/doku.php?id=cases:case008 |

`ahmed-front-geo.dat` is the original surface of the body's rounded front, which `reproduce/build_ahmed_mesh.py`
combines with the rest of the body and its stilts to build the Ahmed-body meshes.

## Terms of use

The NASA files are in the public domain in the United States. The AGARD, McConkey and ERCOFTAC data remain under
the terms of their distributors, and anyone who uses them should cite the original publications listed above,
which are also cited in the paper. The checksums, case definitions and other records written for this repository
are licensed under CC BY 4.0 (`LICENSE-data`).
