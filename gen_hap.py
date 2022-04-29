

from utils import *
from sim_util import *
import heapq

# Global Variables
Frags, Quals = None, None
H, H1, H2 = None, None, None
block_store = None
heap = None
# to maximize ( prod( P(partition | H (Subset(partion)) )) overall partions ) / (C * num_partitions)
# C is a HyperParameter (Should be tuned) 
C = 1.

def get_likelihood_read_given_hap(read, qual, H):
    """
    P(r| H) = ( P(r | h1) +  P(r | h2) ) / 2
    """
    def likelihood_read_given_hap(h):
        """
        param read: read of len Frags[:, subset].shape[1]
        param qual: qual of len Frags[:, subset].shape[1]
        param h: Haplotype of len Frags[:, subset].shape[1] 
        """
        assert read.shape == qual.shape == h.shape
        # Get all reads, i.e. ignnore the indexes having value nan. hea
        indexes = np.argwhere(~np.isnan(read)).flatten()
        likelihood = 1.
        for idx in indexes:
            if read[idx] == h[idx]:
                likelihood *=  (1 - qual[idx])
            else:
                likelihood *= qual[idx]
        
        return likelihood


    l1, l2 = likelihood_read_given_hap(H[0]), likelihood_read_given_hap(H[1])
    
    return (l1 + l2) / 2



class Block:
    def __init__(self, cols):
        self.cols = cols
        read_indexes = []
        
        read_idx = np.zeros(Frags.shape[0], dtype="bool")

        for col in cols:
            read_idx |=  ~np.isnan(Frags[:, col])

        read_indexes = np.nonzero(read_idx)[0]
        
        self.read_indexes = read_indexes        
        # Reads that contain non nan values at atleast one of the cols.
    
    @property
    def hap(self):
        return [H1.take(self.cols), H2.take(self.cols)]

    @property
    def likelihood(self):
        return self.get_likelihood_given_haplotype() 

    def hash(self):
        return hash(tuple(self.cols))

    def is_connected(self, other):
        if len(set(self.read_indexes).intersection(other.read_indexes)):
            return True
        return False


    def get_likelihood_given_haplotype(self):
        cols = self.cols
        read_indexes = self.read_indexes
        reads, quals =  Frags.take(read_indexes, axis=0), Quals.take(read_indexes, axis=0)
        assert reads.shape[1] == quals.shape[1] == Frags.shape[1]

        reads, quals = reads.take(cols, axis=1), quals.take(cols, axis=1)
        assert reads.shape[1] == len(cols)
        hap = self.hap

        likelihood = 1.
        for read, qual in zip(reads, quals):
            # TODO: Make this sum of log(lik)
            likelihood *=  get_likelihood_read_given_hap(read, qual, hap)

        return likelihood    

    def get_likelihood_of_merged(self, other):
        """
        Two blocks can be merged with Haplotypes in same order or the reverse order,
        return the likelihood of the reads for each combination. 
        """
        # Get the value of P( b1 U b2  | h1 h2) / ( P(b1 | h1) * P(b2 | h2)).
        h11, h12 = self.hap
        h21, h22 = other.hap

        hap1 = [np.concatenate([h11, h21]), np.concatenate([h12, h22])]
        hap2 = [np.concatenate([h11, h22]), np.concatenate([h12, h21])]
        cols = list(set(self.cols + other.cols))
        read_indexes = np.unique( np.concatenate([self.read_indexes, other.read_indexes]) )
        reads, quals =  Frags.take(read_indexes, axis=0), Quals.take(read_indexes, axis=0)
        reads, quals = reads.take(cols, axis=1), quals.take(cols, axis=1)

        l1, l2 = 1., 1. 

        for read, qual in zip(reads, quals):
            # TODO: Make this sum of log(lik)
            l1 *= (get_likelihood_read_given_hap(read, qual, hap1))
            l2 *= (get_likelihood_read_given_hap(read, qual, hap2))

        if l1 > l2:
            return l1, hap1, cols

        return l2, hap2, cols

class Edge:
    def __init__(self, lik, b1, b2, hap, cols):
        # The factor by which the likelihood increases if blocks b1 and 
        # b2 are merged.
        self.lik = lik
        self.u = b1
        self.v = b2
        self.hap = hap
        self.cols = cols

    def __lt__(self, other):
        # Override the __lt__ for max heap
        return self.lik > other.lik    


# Data Structure to store blocks 
# O(log(n)) insertion, 0(1) deletion

class BlockStore:
    def __init__(self):
        self.blocks = []
        self.hashd = {}
        self.hashes = set()
    
    def add(self, block):
        # O(1) insert
        if block in self.blocks:
            return
        sz = len(self.blocks)
        self.blocks.append(block)
        self.hashd[block] = sz
        self.hashes.add(block.hash())
    
    def remove(self, block):
        # O(1) remove.
        index = self.hashd.get(block, None)
        if index is None:
            return
        # index = self.blocks.index(bloc)    

        # If present remove the element from the hash
        self.hashes.remove(block.hash())   
        del self.hashd[block]

        # Swap element with the last element as the removal from the 
        # end of the list can be done in O(1) time.
        sz = len(self.blocks)
        last = self.blocks[sz - 1]
        self.blocks[index], self.blocks[sz-1] = self.blocks[sz -1], self.blocks[index]
        del self.blocks[-1]
        self.hashd[last] = index 




def get_blocks_for_reads(fragments_path):
    # Assume all the variant sites to be individual blocks.
    # - two blocks can merge iff 
    #   - the two blocks have a read in common. 
    #   - merging these blocks where the probability of the P( b1 U b2  | h1 h2) / ( P(b1 | h1) * P(b2 | h2))
    import ipdb;ipdb.set_trace()
    global Frags, Quals, H, H1, H2, block_store, heap


    _, Frags, Quals = read_fragments(fragments_path)
    Quals = np.power(10, -0.1* Quals)
    H1, H2 = np.ones(Frags.shape[1]), np.zeros(Frags.shape[1])
    H = [H1, H2]

    block_store = BlockStore()   

    blocks = [Block([i]) for i in range(Frags.shape[1])]
    for block in blocks:
        block_store.add(block)
    
    heap = []

    ipdb.set_trace()
    import pickle
    if os.path.exists("heap.pkl"):
        with open("heap.pkl", "rb") as fd:
            heap = pickle.load(fd)
    else:

        for i in range(len(block_store.blocks)):
            for j in range(len(block_store.blocks)):
                if i == j or not block_store.blocks[i].is_connected(block_store.blocks[j]):
                    # There are no reads connecting the two blocks as of now, 
                    # Hence no need to check whether the two should be merged. 
                    continue
                l, hap, cols = block_store.blocks[i].get_likelihood_of_merged(block_store.blocks[j])
                ed_val = l / (block_store.blocks[i].likelihood * block_store.blocks[j].likelihood)
                edge = Edge(ed_val, block_store.blocks[i], block_store.blocks[j], hap, cols)
                heap.append(edge)
    
    ipdb.set_trace()
    heapq.heapify(heap)
    
    while heap:
        max_edge = heapq.heappop(heap)
        val, b1, b2, hap, cols = max_edge.lik, max_edge.u, max_edge.v, max_edge.hap, max_edge.cols
        # Check both b1 and b2 are present in the block_store
        # if not simply continue as this is not a valid edge anymore.
        if b1.hash() not in block_store.hashes or b2.hash() not in block_store.hashes:
            continue

        num_partitions = len(block_store.blocks)
        # Breaking condition, if this does not hold
        # the overall likelihood decreases.
        if np.log(val) < C*np.log(1 - 1/num_partitions):
            break

        merge_blocks(b1, b2, hap, cols)
        
    import ipdb;ipdb.set_trace()
    
    return block_store     



def merge_blocks(b1, b2, hap, cols):
    # Merge blocks and change the value of the haplotype block. 
    new_block = Block(cols)
    global block_store, H, H1, H2
    try:
        block_store.remove(b1)
        block_store.remove(b2)
    except:
        import ipdb;ipdb.set_trace()
        block_store.remove(b1)
        block_store.remove(b2)  
     
    block_store.add(new_block)
    H1[cols], H2[cols] = hap[0], hap[1]
    H = [H1, H2]
    
    # Go through all the connected blocks of b1 and b2 and
    # add new edges to the heap
    new_block_likelihood = new_block.likelihood 
    for block in block_store.blocks:
        if block.is_connected(new_block) and not len(set(block.cols).intersection(new_block.cols)):
            try:
                nl, nhap, ncols = new_block.get_likelihood_of_merged(block)
            except:
                import ipdb;ipdb.set_trace()
                pass
            ed_val = nl / (new_block_likelihood * block.likelihood)
            edge = Edge(ed_val, new_block, block, nhap, ncols)
            heapq.heappush(heap, edge)
    return        
