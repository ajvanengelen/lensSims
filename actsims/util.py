#-
# util.py
#-
import os,sys, numpy as np
from collections import OrderedDict
from soapack import interfaces as dmint
import logging

class _SeedTracker(object):
    def __init__(self):
        self.CMB     = 0
        self.FG      = 1
        self.PHI     = 2
        self.NOISE   = 3
        self.POISSON = 4
        self.COMPTONY = 5
        self.TILED_NOISE = 6

        #quick-srcfree is maxmally correlated with 15mJy sims
        self.fgdict  = {'15mjy': 0, '100mjy': 1, 'srcfree': 2, 'quick-srcfree':0,'comptony': 3}

        self.dmdict  = {'act_mr3':0,'act_c7v5':1,'planck_hybrid':2,'dr5':3}

    def get_cmb_seed(self, set_idx, sim_idx):
        return (set_idx, 0, self.CMB, sim_idx)

    def get_fg_seed(self, set_idx, sim_idx, fg_type):
        assert(fg_type in self.fgdict.keys())
        return (set_idx, 0, self.FG, sim_idx, self.fgdict[fg_type])

    def get_phi_seed(self, set_idx, sim_idx):
        return (set_idx, 0, self.PHI, sim_idx)

    def get_noise_seed(self, set_idx, sim_idx, data_model, season, patch, array, patch_id=None):
        ret = (set_idx, 0, self.NOISE, sim_idx)
        dm  = data_model
        
        assert(dm.name in self.dmdict.keys())
        sid =  dm.seasons.index(season) if dm.seasons is not None else 0
        pid =  dm.patches.index(patch) if dm.patches is not None else 0
        aid = list(dm.array_freqs.keys()).index(array)
        ret = ret + (self.dmdict[dm.name],sid,pid,aid)
        if patch_id is not None:
            ret = ret + (patch_id,)
        return ret

    def get_poisson_seed(self, set_idx, sim_idx):
        return (set_idx, 0, self.POISSON, sim_idx)

    def get_tiled_noise_seed(self, set_idx, sim_idx, data_model, qid, tile_idx, lowell_seed=False):
        """Return a seed for a tile in a tiled noise simulation scheme. Allows consistent
        seeding for a given simulation set, map number, data model, qid (array), and tile
        number, across users and platforms.

        Parameters
        ----------
        set_idx : int
        sim_idx : int
        data_model : object
            A soapack.interfaces DataModel object
        qid : str or iterable of str
            If simulating 1 array, can pass 1 string or iterable of type string
            and length 1. If simulating correlation between 2 arrays, pass iterable of
            type str and length 2. Iterables are sorted, so order does not matter. Cannot
            correlate more than 2 arrays. 
        tile_idx : int
        lowell_seed : bool
            If two tiling schemes are building one sim, you don't want correlated tiles.
            The second integer in the seed tuple will be 1 if True.

        Returns
        -------
        tuple of int
            Seed to be passed to np.random.seed

        Example
        -------
        >>> from actsims import util as u
        >>> u.seed_tracker.get_tiled_noise_seed(3,963,u.dmint.DR5(),'s18_03',7_034)
        >>> (3, 0, 6, 963, 3, 8326, 0, 7034)
        >>> u.seed_tracker.get_tiled_noise_seed(3,963,u.dmint.DR5(),['s18_04','s18_03'],7_034)
        >>> (3, 0, 6, 963, 3, 8326, 2839, 7034)

        """
        ret = (set_idx, int(lowell_seed), self.TILED_NOISE, sim_idx)
        dm = data_model
        qid = np.sort(np.atleast_1d(qid)) # sorted qids
        assert len(qid) <= 2, f'Can only seed for correlation of up to 2 arrays; {len(qid)} passed'

        assert(dm.name in self.dmdict.keys())
        dm_idx = self.dmdict[dm.name]
        
        if len(qid) == 1:
            qid_idx = (dmint.arrays(qid[0], 'hash'), 0)
        else:
            qid_idx = tuple(dmint.arrays(q, 'hash') for q in qid)
        return ret + (dm_idx,) + qid_idx + (tile_idx,)
seed_tracker = _SeedTracker()


#########################


def is_empty(s):
    '''
        if s is either '' or None, this function returns True. Otherwise, false.
    '''
    return True if (s is None) or (len(s) == 0) else False

def create_dict(*idxes):
    '''
        create nested dictionary with the given idxes
    '''

    height  = len(idxes)
    output = {}
    
    stack  = []
    stack.append(output)

    for depth in range(height):
        stack_temp = []
        while len(stack) > 0:
            cur_elmt = stack.pop()
            for idx in idxes[depth]:
                cur_elmt[idx] = {}
                stack_temp.append(cur_elmt[idx])
        stack = stack_temp

    return output

def get_from_dict(nested_dict, keys, safe=True):
    if not type(keys) is tuple: keys = (keys,)
    if safe and not has_key(nested_dict, keys): return None

    if(len(keys) > 1):
        return get_from_nested_dict(nested_dict[keys[0]], keys[1:], False)
    else:
        return nested_dict[keys[0]]


def has_key(nested_dict, keys):
    ''' search through nested dictionary to fine the elements '''
    if not type(keys) is tuple: keys = (keys,)
    if not type(nested_dict) == dict: return False

    if(len(keys) > 1):
        has_it = keys[0] in nested_dict
        return has_key(nested_dict[keys[0]], keys[1:]) if has_it else False
    else:
        return keys[0] in nested_dict

def get_num_leaves(nested_dict):
    ''' search how many elements are in nested dictionary '''
    count = 0
    for key in nested_dict.keys():
        if isinstance(nested_dict[key], dict):
            count += get_num_elements(nested_dict[key])
        else:
            count += 1
    return count


# decorators
def static_vars(**kwargs):
    '''https://stackoverflow.com/questions/279561/what-is-the-python-equivalent-of-static-variables-inside-a-function'''
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

def create_dir(path_to_dir):
    ''' check wether the directory already exists. if not, create it '''
    exit = 0       # exit code

    if not os.path.isdir(path_to_dir):
        os.makedirs(path_to_dir)
        exit = 0
    else:
        exit = 1

    return exit

# took it from quicklens/util.py
class memorize(object):
    """ a simple memoize decorator (http://en.wikipedia.org/wiki/Memoization) """
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if args in self.cache:
            logging.info("returning from cache.")
            return self.cache[args]
        else:
            v = self.func(*args)
            self.cache[args] = v
            return v

def mkdir(dirpath,comm=None):
    from mpi4py import MPI
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    exists = os.path.exists(dirpath)
    comm.Barrier()
    if comm.Get_rank()==0: 
        if not (exists):
            os.makedirs(dirpath)
    return exists

def mpi_distribute(num_tasks,avail_cores,allow_empty=False):
    # copied to mapsims.convert_noise_templates
    if not(allow_empty): assert avail_cores<=num_tasks
    min_each, rem = divmod(num_tasks,avail_cores)
    num_each = np.array([min_each]*avail_cores) # first distribute equally
    if rem>0: num_each[-rem:] += 1  # add the remainder to the last set of cores (so that rank 0 never gets extra jobs)

    task_range = list(range(num_tasks)) # the full range of tasks
    cumul = np.cumsum(num_each).tolist() # the end indices for each task
    task_dist = [task_range[x:y] for x,y in zip([0]+cumul[:-1],cumul)] # a list containing the tasks for each core
    assert sum(num_each)==num_tasks
    assert len(num_each)==avail_cores
    assert len(task_dist)==avail_cores
    return num_each,task_dist


def distribute(njobs,verbose=True,**kwargs):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    numcores = comm.Get_size()
    num_each,each_tasks = mpi_distribute(njobs,numcores,**kwargs)
    if rank==0: logging.info(f"MPI: At most {max(num_each)} tasks...")
    my_tasks = each_tasks[rank]
    return comm,rank,my_tasks

def config_from_yaml(filename):
    import yaml
    with open(filename) as f:
        config = yaml.safe_load(f)
    return config

"""
I'll probably be moving these git info functions to a pipelining package later
COPIED INTO PYFISHER
"""
def pretty_info(info):
    name = info['package'] if info['package'] is not None else info['path']
    pstr = f'\n{name}'
    pstr = pstr + '\n'+''.join(["=" for x in range(len(name))])
    for key in info.keys():
        if key=='package': continue
        pstr = pstr + '\n' + f'\t{key:<10}{str(info[key]):<40}'
    return pstr

def get_info(package=None,path=None,validate=True):
    import git
    import importlib
    info = {}
    if package is None:
        assert path is not None, "One of package or path must be specified."
        path = os.path.dirname(path)
        version = None
    else:
        mod = importlib.import_module(package)
        try:
            version = mod.__version__
        except AttributeError:
            version = None
        path = mod.__file__
        path = os.path.dirname(path)
    info['package'] = package
    info['path'] = path
    info['version'] = version
    try:
        repo = git.Repo(path,search_parent_directories=True)
        is_git = True
    except git.exc.InvalidGitRepositoryError:
        is_git = False
    info['is_git'] = is_git
    if is_git:
        chash = str(repo.head.commit)
        untracked = len(repo.untracked_files)>0
        changes = len(repo.index.diff(None))>0
        branch = str(repo.active_branch)
        info['hash'] = chash
        info['untracked'] = untracked
        info['changes'] = changes
        info['branch'] = branch
    else:
        if validate:
            assert version is not None
            assert 'site-packages' in path
    return info
    
