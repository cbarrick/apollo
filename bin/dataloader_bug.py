from threading import Lock
from torch.utils.data import Dataset, DataLoader
import xarray as xr

class MyDataset(Dataset):
    '''A simple torch Dataset wrapping xr.DataArray'''
    def __init__(self, ar, batch_dim):
        self.ar = ar
        self.batch_dim = batch_dim

    def __len__(self):
        return len(self.ar[self.batch_dim])

    def __getitem__(self, idx):
        return self.ar[{self.batch_dim: idx}].values

# Open the dataset with xarray and select the feature to work with.
data = ['NAM-NMM/nam.20161111/nam.t00z.awphys.tm00.nc', 'NAM-NMM/nam.20161211/nam.t00z.awphys.tm00.nc']
data = xr.open_mfdataset(data, lock=False)
data = data['TMP_SFC']

# If I load the data into main memory, the deadlock does not happen
#data.load()

# Trying to load the data with num_workers > 0 results in deadlock.
ds = MyDataset(data, batch_dim='reftime')
dl = DataLoader(ds, num_workers=1)
i = iter(dl)
next(i)
