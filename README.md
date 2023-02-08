# C3C4VeinSpacing

Program: leafDevSimulator_v20221124.py

Developer: Tiangen Chang (changtiangen@gmail.com)


## Aims and model variables

### This program aims to simulating typical monocot vein patterning with different levels of SHR-SCR and auxin, where:

   SHR-SCR module will 1) promotes MC growth and division and 2) inhibits vein formation by inhibiting Auxin synthesis;

   SG (substrate for growth) comes from vein and diffuse to outer layers to support cell growth and division;

   Auxin produces by ground meristem cells and diffuse to sinks (veins) with conc. gradient. It will 1) promotes vein formation and 2) inhibits MC growth and division.


### Variables: (SHR, SCR, Auxin, SG)
   
   synthesis rate
   
   degradation rate
   
   diffuse rate


## Run commands 

Simulate Rice_WT: python leafDevSimulator_v20221124.py 3 3 0.5

Simulate Rice_shr: python leafDevSimulator_v20221124.py 1.5 3 0.5

Simulate Rice_scr: python leafDevSimulator_v20221124.py 3 0.6 0.5

Simulate Rice_SHR_OE: python leafDevSimulator_v20221124.py 9 3 0.5

Simulate Rice_addAuxin: python leafDevSimulator_v20221124.py 3 3 1.5

Simulate Rice_SHR_OE_addAuxin: python leafDevSimulator_v20221124.py 9 3 1.5

Simulate Maize_WT: python leafDevSimulator_v20221124.py 6 6 2.5

Simulate Maize_shr: python leafDevSimulator_v20221124.py 1.5 6 2.5

Simulate Maize_scr: python leafDevSimulator_v20221124.py 6 1.2 2.5

Simulate Maize_SHR_OE: python leafDevSimulator_v20221124.py 30 6 2.5
