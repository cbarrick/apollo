# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:52:50 2019

@author: fwmaier
"""

from pathlib import Path
import pkg_resources as pkgr


def get_asset_string(resource):
    """If the user has specified a file with a path that exists, 
    then use it. Otherwise, check in the templates directory.  
    """
    try:
        path = Path(resource)
        if path.exists and path.is_file():
            with open(resource, 'r') as f:
                return f.read()
    except:
        pass
    return pkgr.resource_string('apollo', resource).decode("utf-8")


