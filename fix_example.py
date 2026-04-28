import sys

with open('examples/ml_color_bias_diagnostics.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'mag_H = cat["MAG_H"]' in line:
        new_lines.append('import numpy as np\n')
        new_lines.append('def flux_to_mag(flux): return -2.5 * np.log10(flux) + 23.9\n')
        new_lines.append('mag_H = flux_to_mag(cat["flux_h_templfit"])\n')
    elif 'mag_VIS = cat["MAG_VIS"]' in line:
        new_lines.append('mag_VIS = flux_to_mag(cat["flux_vis_psf"])\n')
    elif 'mag_Y = cat["MAG_Y"]' in line:
        new_lines.append('mag_Y = flux_to_mag(cat["flux_y_templfit"])\n')
    elif 'mag_J = cat["MAG_J"]' in line:
        new_lines.append('mag_J = flux_to_mag(cat["flux_j_templfit"])\n')
    elif 'logM_SBI = cat["LOGM_SBI"]' in line:
        new_lines.append('logM_SBI = cat["mass_med"] # SBI mock replacement\n')
    elif 'logM_CWeb = cat["LOGM_CWEB"]' in line:
        new_lines.append('logM_CWeb = cat["mass_med"] # COSMOS-Web replacement\n')
    elif 'logM_CWeb = cat["LOGM_COSMOS"]' in line:
        new_lines.append('logM_CWeb = cat["mass_med"] # COSMOS-Web line 2 replacement\n')
    elif 'z_CWeb = cat["Z_CWEB"]' in line or 'z = cat["REDSHIFT"]' in line:
        new_lines.append('z = cat["zfinal"] # COSMOS-Web redshift replacement\n')
    else:
        new_lines.append(line)

with open('examples/ml_color_bias_diagnostics.py', 'w') as f:
    f.writelines(new_lines)
