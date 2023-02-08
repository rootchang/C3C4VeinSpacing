import copy
import matplotlib.pylab as plt
import numpy as np
import sys

end_time = 1000000 # total simulation time: pseudo-seconds, ~10 pseudo-days
EPSILON = 0.000001 # numeric small number
N_sigma = 0.05  # stochastic noise
np.random.seed(1)

#synthesis and degradation
deg_coef = 20
C3_SHR_coef = float(sys.argv[1]) # C3: 3; C4: 6
C3_SCR_coef = float(sys.argv[2]) # C3: 3; C4: 6
C3_auxin_coef = float(sys.argv[3]) # C3: 0.5; C4: 2.5

#SHR
k_deg_S = deg_coef*2e-06  # 20%/d degradation of SHR --> max conc. = 500
vm_syn_S_V = C3_SHR_coef*5*1e-3 # ~100 unit per day in vein
vm_syn_S_M = C3_SHR_coef*10*1e-5 # much lower exp. in M
cond_S = 0.5*1e-4 # coef. delta_conc per second
#SCR
k_deg_C = deg_coef*2e-06  # 20%/d degradation
vm_syn_C_V = C3_SCR_coef*2*1e-3 # SCR expressed in both BS and M
vm_syn_C_M = C3_SCR_coef*5*1e-3 # ~100 unit per day
c0_C_SC = 0.0 # when [S-C] = 0, vm = a0 + 1/(1+exp(a1*[SC]+a2)) = 0.27
c1_C_SC = -0.2 # when [S-C] = 40, vm = 0.75
c2_C_SC = 2 # when [S-C] = 70, vm = 1.2
cond_C = 0 #1e-4 # coef. delta_conc per second
#SHR-SCR
k_deg_SC = deg_coef*2e-06  # 20%/d degradation
km_SC_S = 40
km_SC_C = 40
vm_syn_SC_V = 5*1e-3 # ~100 unit per day
vm_syn_SC_M = 5*1e-3 # ~100 unit per day
cond_SC = 0
#Auxin
k_deg_auxin = 0.1*deg_coef*2e-06  # 20%/d degradation
vm_syn_auxin_V = 0
vm_syn_auxin_M = C3_auxin_coef*1e-3 # 1 for C4
b0_auxin_SC = 0.25 # when [S-C] = 0, vm = b0 + 1/(1+exp(b1*[SC]+b2)) = 1.25
b1_auxin_SC = 0.2 # when [S-C] = 20, vm = 0.7
b2_auxin_SC = -1 # when [S-C] = 40, vm = 0.3
cond_auxin = 0.2*1e-4 # coef. delta_conc per second
#SG (substrate for growth)
cond_SG = 0.2*1e-4 # coef. delta_conc per second

#cell growth
cell_Vol = 1 # leaf M cell volume (1 unit). Mature vein: 4??
cell_Vol_upper = 2  # critical volume for cell division
vm_Vol_growth = 5*8e-6  # volume growth coef./second -- > 100%/d volume growth/day
km_growth_SG = 20
SG2growth_coef = 10 # 10 SGs will be used for 1 volume growth
SG2maintain_coef = 100*1e-6 # 1 SGs will be used for 1 volume maintain for one day
#SC --> growth
a0_growth_SC = 0 # when [S-C] = 0, vm = a0 + 1/(1+exp(a1*[SC]+a2)) = 0.27
a1_growth_SC = -0.1 # when [S-C] = 40, vm = 0.75
a2_growth_SC = 2 # when [S-C] = 70, vm = 1.2
#Auxin --| growth
d0_growth_auxin = 0.25 # when [Auxin] = 0, vm = b0 + 1/(1+exp(b1*[SC]+b2)) = 1.25
d1_growth_auxin = 0.2 # when [Auxin] = 20, vm = 0.7
d2_growth_auxin = -1 # when [Auxin] = 40, vm = 0.3

#vein formation
auxin_upper = 30  # critical diffV concentration for procambium initiation

#### initial concentrations.
c_S_V = 0
c_C_V = 0
c_SC_V = 0
c_auxin_V = 0 # constant auxin conc. in vein
c_SG_V = 50 # constant SG conc. in vein

c_S_M = 0
c_C_M = 0
c_SC_M = 0
c_auxin_M = 0
c_SG_M = 20

cell_list = []
#### composition of elements
# name
# volume
# conc: SHR, SCR, S-C, auxin, SG
# vmax_and_cond: vm_S, vm_C, vm_SC, vm_auxin, vm_growth, cond_coef (for S, C, SC, auxin, SG)
rd1 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd2 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd3 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd4 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd5 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd6 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
cell_list.append(['V', 1, [c_S_V,c_C_V,c_SC_V,c_auxin_V,c_SG_V], [vm_syn_S_V*rd1, vm_syn_C_V*rd2, vm_syn_SC_V*rd3, vm_syn_auxin_V*rd4, vm_Vol_growth*rd5, 1*rd6]])  # left Vein
rd1 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd2 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd3 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd4 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd5 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd6 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
cell_list.append(['M', 1, [c_S_M,c_C_M,c_SC_M,c_auxin_M,c_SG_M], [vm_syn_S_M*rd1, vm_syn_C_M*rd2, vm_syn_SC_M*rd3, vm_syn_auxin_M*rd4, vm_Vol_growth*rd5, 1*rd6]])  # M1
rd1 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd2 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd3 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd4 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd5 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd6 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
cell_list.append(['M', 1, [c_S_M,c_C_M,c_SC_M,c_auxin_M,c_SG_M], [vm_syn_S_M*rd1, vm_syn_C_M*rd2, vm_syn_SC_M*rd3, vm_syn_auxin_M*rd4, vm_Vol_growth*rd5, 1*rd6]])  # M2
rd1 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd2 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd3 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd4 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd5 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd6 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
cell_list.append(['M', 1, [c_S_M,c_C_M,c_SC_M,c_auxin_M,c_SG_M], [vm_syn_S_M*rd1, vm_syn_C_M*rd2, vm_syn_SC_M*rd3, vm_syn_auxin_M*rd4, vm_Vol_growth*rd5, 1*rd6]])  # M3
rd1 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd2 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd3 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd4 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd5 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd6 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
cell_list.append(['M', 1, [c_S_M,c_C_M,c_SC_M,c_auxin_M,c_SG_M], [vm_syn_S_M*rd1, vm_syn_C_M*rd2, vm_syn_SC_M*rd3, vm_syn_auxin_M*rd4, vm_Vol_growth*rd5, 1*rd6]])  # M4
rd1 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd2 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd3 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd4 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd5 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd6 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
cell_list.append(['M', 1, [c_S_M,c_C_M,c_SC_M,c_auxin_M,c_SG_M], [vm_syn_S_M*rd1, vm_syn_C_M*rd2, vm_syn_SC_M*rd3, vm_syn_auxin_M*rd4, vm_Vol_growth*rd5, 1*rd6]])  # M5
rd1 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd2 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd3 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd4 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd5 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
rd6 = (1+min(0.9,max(-0.9,np.random.normal(scale=N_sigma))))
cell_list.append(['V', 1, [c_S_V,c_C_V,c_SC_V,c_auxin_V,c_SG_V], [vm_syn_S_V*rd1, vm_syn_C_V*rd2, vm_syn_SC_V*rd3, vm_syn_auxin_V*rd4, vm_Vol_growth*rd5, 1*rd6]])  # right Vein

# record conc change with time for plot
conc_S_time = []
conc_C_time = []
conc_SC_time = []
conc_auxin_time = []
conc_SG_time = []
record_step = 1000

real_simulation = 1 # if = 0, neither cells divide nor veins form

# simulation
for t_i in range(end_time):
    cellNUM = len(cell_list)
    cell_list_new = [cell_list[0] for _ in range(cellNUM*2)]
    pos_i = 0
    for c_i in range(cellNUM):
        new_vein = 0
        cn_temp = cell_list[c_i][0]
        cell_Vol_temp = cell_list[c_i][1]
        c_SHR_temp, c_SCR_temp, c_SC_temp, c_auxin_temp, c_SG_temp = cell_list[c_i][2]
        vm_S_temp, vm_C_temp, vm_SC_temp, vm_auxin_temp, vm_growth_temp, cond_coef_temp = cell_list[c_i][3]
        cond_S_temp, cond_C_temp, cond_SC_temp, cond_auxin_temp, cond_SG_temp = cond_coef_temp*np.array([cond_S, cond_C, cond_SC, cond_auxin, cond_SG])

        c_i_left = c_i - 1
        c_SHR_left, c_SCR_left, c_SC_left, c_auxin_left, c_SG_left = cell_list[c_i_left][2]
        vm_S_left, vm_C_left, vm_SC_left, vm_auxin_left, vm_growth_left, cond_coef_left = cell_list[c_i_left][3]
        cond_S_left, cond_C_left, cond_SC_left, cond_auxin_left, cond_SG_left = cond_coef_left*np.array([cond_S, cond_C, cond_SC, cond_auxin, cond_SG])
        cond_S_left = min(cond_S_left,cond_S_temp)
        cond_C_left = min(cond_C_left, cond_C_temp)
        cond_SC_left = min(cond_SC_left, cond_SC_temp)
        cond_auxin_left = min(cond_auxin_left, cond_auxin_temp)
        cond_SG_left = min(cond_SG_left, cond_SG_temp)

        c_i_right = c_i + 1
        if c_i_right == cellNUM:
            c_i_right = 0
        c_SHR_right, c_SCR_right, c_SC_right, c_auxin_right, c_SG_right = cell_list[c_i_right][2]
        vm_S_right, vm_C_right, vm_SC_right, vm_auxin_right, vm_growth_right, cond_coef_right = cell_list[c_i_right][3]
        cond_S_right, cond_C_right, cond_SC_right, cond_auxin_right, cond_SG_right = cond_coef_right*np.array([cond_S, cond_C, cond_SC, cond_auxin, cond_SG])
        cond_S_right = min(cond_S_right,cond_S_temp)
        cond_C_right = min(cond_C_right, cond_C_temp)
        cond_SC_right = min(cond_SC_right, cond_SC_temp)
        cond_auxin_right = min(cond_auxin_right, cond_auxin_temp)
        cond_SG_right = min(cond_SG_right, cond_SG_temp)

        v_SHR_deg = c_SHR_temp*cell_Vol_temp*k_deg_S
        v_SHR_left_in = (c_SHR_left - c_SHR_temp)*cond_S_left
        v_SHR_right_in = (c_SHR_right - c_SHR_temp)*cond_S_right
        if (c_SHR_temp < EPSILON) or (c_SCR_temp < EPSILON):
            v_SC_syn = 0
        else:
            v_SC_syn = vm_SC_temp * 1/(1+km_SC_S/c_SHR_temp+km_SC_C/c_SCR_temp)
        c_SHR_temp = c_SHR_temp + (vm_S_temp - v_SHR_deg + v_SHR_left_in + v_SHR_right_in - v_SC_syn)/cell_Vol_temp

        v_SCR_syn = vm_C_temp*(c0_C_SC+1/(1+np.exp(c1_C_SC*c_SC_temp+c2_C_SC)))
        v_SCR_deg = c_SCR_temp * cell_Vol_temp * k_deg_C
        v_SCR_left_in = (c_SCR_left - c_SCR_temp) * cond_C_left
        v_SCR_right_in = (c_SCR_right - c_SCR_temp) * cond_C_right
        c_SCR_temp = c_SCR_temp + (v_SCR_syn - v_SCR_deg + v_SCR_left_in + v_SCR_right_in - v_SC_syn)/cell_Vol_temp

        v_SC_deg = c_SC_temp * cell_Vol_temp * k_deg_SC
        v_SC_left_in = (c_SC_left - c_SC_temp) * cond_SC_left
        v_SC_right_in = (c_SC_right - c_SC_temp) * cond_SC_right
        c_SC_temp = c_SC_temp + (v_SC_syn - v_SC_deg + v_SC_left_in + v_SC_right_in)/cell_Vol_temp

        if cn_temp != 'V':
            v_auxin_syn = vm_auxin_temp * (b0_auxin_SC + 1 / (1 + np.exp(b1_auxin_SC * c_SC_temp + b2_auxin_SC)))
            v_auxin_deg = c_auxin_temp * cell_Vol_temp * k_deg_auxin
            v_auxin_left_in = (c_auxin_left - c_auxin_temp) * cond_auxin_left
            v_auxin_right_in = (c_auxin_right - c_auxin_temp) * cond_auxin_right
            c_auxin_temp = c_auxin_temp + (v_auxin_syn - v_auxin_deg + v_auxin_left_in + v_auxin_right_in)/cell_Vol_temp
            if real_simulation:
                if c_auxin_temp > auxin_upper:
                    cn_temp = 'V'
                    new_vein = 1
                else:
                    v_growth = c_SG_temp / (km_growth_SG + c_SG_temp) * (a0_growth_SC + 1 / (1 + np.exp(a1_growth_SC * c_SC_temp + a2_growth_SC))) * (d0_growth_auxin + 1 / (1 + np.exp(d1_growth_auxin * c_SC_temp + d2_growth_auxin))) * vm_growth_temp * cell_Vol_temp
                    v_use = v_growth
                    v_SG_left_in = (c_SG_left - c_SG_temp) * cond_SG_left
                    v_SG_right_in = (c_SG_right - c_SG_temp) * cond_SG_right
                    c_SG_temp = c_SG_temp + ( - v_use + v_SG_left_in + v_SG_right_in)/cell_Vol_temp
                    # conc. will decrease with enlarged cell volume
                    cell_Vol_temp = cell_Vol_temp + v_growth
                    dilute_coef= cell_Vol_temp/(cell_Vol_temp+v_growth)
                    c_SHR_temp, c_SCR_temp, c_SC_temp, c_auxin_temp, c_SG_temp = dilute_coef*np.array([c_SHR_temp, c_SCR_temp, c_SC_temp, c_auxin_temp, c_SG_temp])
            else:
                v_growth = c_SG_temp / (km_growth_SG + c_SG_temp) * (a0_growth_SC + 1 / (1 + np.exp(a1_growth_SC * c_SC_temp + a2_growth_SC))) * (d0_growth_auxin + 1 / (1 + np.exp(d1_growth_auxin * c_SC_temp + d2_growth_auxin))) * vm_growth_temp * cell_Vol_temp
                v_use = SG2growth_coef*v_growth + SG2maintain_coef*cell_Vol_temp
                v_SG_left_in = (c_SG_left - c_SG_temp) * cond_SG_left
                v_SG_right_in = (c_SG_right - c_SG_temp) * cond_SG_right
                c_SG_temp = max(0, c_SG_temp + (- v_use + v_SG_left_in + v_SG_right_in) / cell_Vol_temp)
        if real_simulation:
            if (cn_temp != 'V'):
                if cell_Vol_temp >= 2: # new Ms formation
                    cell_Vol_temp = cell_Vol_temp/2
                    rd1 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd2 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd3 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd4 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd5 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd6 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))

                    cell_list_new[pos_i] = [cn_temp, cell_Vol_temp,[c_SHR_temp, c_SCR_temp, c_SC_temp, c_auxin_temp, c_SG_temp], cell_list[c_i][3]]
                    cell_list_new[pos_i][3]=[vm_syn_S_M * rd1, vm_syn_C_M * rd2, vm_syn_SC_M * rd3, vm_syn_auxin_M * rd4, vm_Vol_growth * rd5, 1 * rd6]
                    rd1 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd2 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd3 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd4 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd5 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    rd6 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                    cell_list_new[pos_i+1] = [cn_temp, cell_Vol_temp,[c_SHR_temp, c_SCR_temp, c_SC_temp, c_auxin_temp, c_SG_temp], cell_list[c_i][3]]
                    cell_list_new[pos_i+1][3] = [vm_syn_S_M * rd1, vm_syn_C_M * rd2, vm_syn_SC_M * rd3, vm_syn_auxin_M * rd4,vm_Vol_growth * rd5, 1 * rd6]
                    pos_i += 2
                else: # old M
                    cell_list_new[pos_i] = [cn_temp, cell_Vol_temp,[c_SHR_temp, c_SCR_temp, c_SC_temp, c_auxin_temp, c_SG_temp],cell_list[c_i][3]]
                    pos_i += 1
            elif new_vein > 0.5: # new vein formation
                rd1 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                rd2 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                rd3 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                rd4 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                rd5 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                rd6 = (1 + min(0.9, max(-0.9, np.random.normal(scale=N_sigma))))
                cell_list_new[pos_i] = ['V', 1, [c_SHR_temp, c_SCR_temp, c_SC_temp, c_auxin_V, c_SG_V],[vm_syn_S_V * rd1, vm_syn_C_V * rd2, vm_syn_SC_V * rd3, vm_syn_auxin_V * rd4,vm_Vol_growth * rd5, 1 * rd6]]  # left Vein
                pos_i += 1
            else: # old vein
                cell_list_new[pos_i] = [cn_temp, cell_Vol_temp,[c_SHR_temp, c_SCR_temp, c_SC_temp, c_auxin_temp, c_SG_temp], cell_list[c_i][3]]
                pos_i += 1
        else: # no growth (for debugging)
            cell_list_new[pos_i] = [cn_temp, cell_Vol_temp,[c_SHR_temp, c_SCR_temp, c_SC_temp, c_auxin_temp, c_SG_temp], cell_list[c_i][3]]
            pos_i += 1
    cell_list_new = cell_list_new[0:pos_i]
    cell_list = copy.deepcopy(cell_list_new)
    if not t_i%record_step:
        #print('Time: %7d\t'%t_i, [[c[0],round(c[2][0],0),round(c[2][1],0),round(c[2][2],0)] for c in cell_list])
        content = ''.join([c[0] for c in cell_list])
        content = content.split('V')[1:-1]
        MCL_list = [len(c) for c in content]
        content = ' '.join([str(c) for c in MCL_list])
        print('Time: %7d\t' % t_i, content, round(np.mean(MCL_list),2))
        if not real_simulation:
            all_concs = [c[2] for c in cell_list]
            SHR_conc_temp, SCR_conc_temp, SC_conc_temp, auxin_conc_temp, SG_conc_temp = [list(c) for c in zip(*all_concs)]
            conc_S_time.append(SHR_conc_temp)
            conc_C_time.append(SCR_conc_temp)
            conc_SC_time.append(SC_conc_temp)
            conc_auxin_time.append(auxin_conc_temp)
            conc_SG_time.append(SG_conc_temp)

# plot conc. dynamics with time
if not real_simulation:
    ax = [0 for _ in range(35)]
    output_fig = 'conc_VS_time.png'
    fig1, ((ax[0],ax[1],ax[2],ax[3],ax[4],ax[5],ax[6]),(ax[7],ax[8],ax[9],ax[10],ax[11],ax[12],ax[13]),
           (ax[14],ax[15],ax[16],ax[17],ax[18],ax[19],ax[20]),(ax[21],ax[22],ax[23],ax[24],ax[25],ax[26],ax[27]),
           (ax[28],ax[29],ax[30],ax[31],ax[32],ax[33],ax[34])) = plt.subplots(5, 7, figsize=(15, 15))  # figsize=(6, 8)
    plt.rcParams['font.size'] = 12
    plt.subplots_adjust(left=0.1, bottom=0.11, right=0.98, top=0.98, wspace=0.35, hspace=0.35)

    time_x = range(0,end_time,record_step)
    y_max = max(1, max(sum(conc_S_time,[])))
    conc_S_time = list(zip(*conc_S_time))
    for i in range(7):
        ax[i].plot(time_x,conc_S_time[i],'k-',linewidth=1)
        ax[i].set_ylim([0,y_max*1.2])
        ax[i].set_xlim([0,end_time])
        ax[i].set_ylabel('SHR')
        ax[i].spines.right.set_visible(False)
        ax[i].spines.top.set_visible(False)
    y_max = max(1, max(sum(conc_C_time,[])))
    conc_C_time = list(zip(*conc_C_time))
    for i in range(7):
        ax[7+i].plot(time_x,conc_C_time[i],'r-',linewidth=1)
        ax[7+i].set_ylim([0,y_max*1.2])
        ax[7+i].set_xlim([0,end_time])
        ax[7+i].set_ylabel('SCR')
        ax[7+i].spines.right.set_visible(False)
        ax[7+i].spines.top.set_visible(False)
    y_max = max(1, max(sum(conc_SC_time,[])))
    conc_SC_time = list(zip(*conc_SC_time))
    for i in range(7):
        ax[14+i].plot(time_x,conc_SC_time[i],'g-',linewidth=1)
        ax[14+i].set_ylim([0,y_max*1.2])
        ax[14+i].set_xlim([0,end_time])
        ax[14+i].set_ylabel('SHR-SCR')
        ax[14+i].spines.right.set_visible(False)
        ax[14+i].spines.top.set_visible(False)
    y_max = max(1, max(sum(conc_auxin_time,[])))
    conc_auxin_time = list(zip(*conc_auxin_time))
    for i in range(7):
        ax[21+i].plot(time_x,conc_auxin_time[i],'b-',linewidth=1)
        ax[21+i].set_ylim([0,y_max*1.2])
        ax[21+i].set_xlim([0,end_time])
        ax[21+i].set_ylabel('Auxin')
        ax[21+i].spines.right.set_visible(False)
        ax[21+i].spines.top.set_visible(False)
    y_max = max(1, max(sum(conc_SG_time,[])))
    conc_SG_time = list(zip(*conc_SG_time))
    for i in range(7):
        ax[28+i].plot(time_x,conc_SG_time[i],'c-',linewidth=1)
        ax[28+i].set_ylim([0,y_max*1.2])
        ax[28+i].set_xlim([0,end_time])
        ax[28+i].set_ylabel('SG')
        ax[28+i].spines.right.set_visible(False)
        ax[28+i].spines.top.set_visible(False)
    fig1.savefig(output_fig, dpi=300)
    plt.close()