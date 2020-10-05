import os
import tempfile

import numpy as np
from hashlib import sha1

import numba

from sqlalchemy import Column,String,ForeignKey,Integer,Date
from sqlalchemy.orm import relationship

from orm import Base
from slicing import shift_slice
from geometry import containedby
STORAGEPATH = "store"
BASEEXTENT = (720,360)

def storagepath(fid):
    if os.getenv("TDB_TEST"):
        STORAGEPATH = "/tmp"
    return os.path.join(STORAGEPATH,fid)

def tensorid(tensor):
    return sha1(tensor).hexdigest()

def missing(*args,**kwargs):
    x = np.zeros(*args,**kwargs)
    x[:] = np.nan
    return x

def compileTensors(tensors,extent,x=None,y=None,z=None):

    def inExtent(tensor):
        t = tensor.get(x,y,z)
        baseIdx = [shift_slice(slice(*i),step) for i,step in zip((x,y,z),tensor.anchors())]
        base = missing(extent)

        bx,by,bz = baseIdx
        base[bx,by,bz] = t
        return t 

    return np.array([inExtent(t) for t in tensors])

class Tensor(Base):
    __tablename__ = "tensor"
    id = Column(String(32),primary_key=True)
    feature_id = Column(ForeignKey("feature.id"))
    feature = relationship("Feature")

    x_anchor = Column(Integer)
    y_anchor = Column(Integer)
    z_anchor = Column(Integer)

    timestamp = Column(Date)

    def path(self):
        return storagepath(self.id)

    def __getitem__(self,idx):
        """
        Allows slicing the tensor on disk, with slices shifted relative to the
        tensor anchor.
        """
        #idx = [*zip(idx,(self.x_anchor,self.y_anchor,self.z_anchor))]
        #idx = tuple((shift_slice(sl,-step) for sl,step in idx))

        mmap = self._memmap()
        return np.array(mmap[idx])

    def anchors(self):
        return self.x_anchor,self.y_anchor,self.z_anchor

    def _memmap(self):
        return np.load(self.path(),mmap_mode="r")

    def get(self,x=None,y=None,z=None):
        indices = [x,y,z]
        mmap = self._memmap()

        for idx,i in enumerate(indices):
            if i is None:
                indices[idx] = (0,mmap.shape[idx])

        #indices = zip((np.array(i) for i in indices),(self.x_anchor,self.y_anchor,self.z_anchor))
        #indices = (i-sh for i,sh in indices)
        indices = (slice(sta,sto) for sta,sto in indices)

        x,y,z = indices

        return np.array(mmap[x,y,z]).squeeze()

    @classmethod
    def new(cls,tensor,x=0,y=0,z=0):
        """
        Create a new tensor instance, serializing the tensor on disk.

        x,y and z can be given to specify the origo for each of the axes.  This
        is done to allow for meta-slicing between tensors, relating all tensors
        to a common index while avoiding storing useless information to pad
        them out. The padding is added when "compiling" a list of tensors into
        a common extent.
        """
        assert len(tensor.shape) == 3

        id = tensorid(tensor)

        instance = cls(
            id = id,
            x_anchor=x,
            y_anchor=y,
            z_anchor=z,
        )

        with open(instance.path(),"wb") as f:
            np.save(f,tensor)

        return instance
