import h5py
import matplotlib.pyplot as plt

# Path to atlas file
atlas_path = 'library/atlas_obs_euclid_north_validate_20000_Nparam_2.dbatlas'

with h5py.File(atlas_path, 'r') as f:
    mstar = f['data/"mstar"'][:]
    zval = f['data/"zval"'][:]

plt.figure(figsize=(6,4))
plt.hexbin(zval, mstar, gridsize=100, cmap='viridis', bins='log')
plt.xlabel('Redshift (z)')
plt.ylabel('log(M/Msun)')
plt.title('Atlas: Stellar Mass vs Redshift')
plt.colorbar(label='log(N)')
plt.tight_layout()
plt.savefig('atlas_mass_vs_z.png', dpi=150)
plt.show()
print('Plot saved as atlas_mass_vs_z.png')
