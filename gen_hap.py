


from utils import *
from sim_util import *
import heapq
import warnings

warnings.filterwarnings("error")

def breakpoint():
    import ipdb;ipdb.set_trace()
    pass

# Global Variables
Frags, Quals = None, None
H, H1, H2 = None, None, None
block_store = None
cluster_store = None
heap = None
Blocks = []
INF = 4e9

# to maximize ( prod( P(partition | H (Subset(partion)) )) overall partions ) / (C * num_partitions)
# C is a HyperParameter (Should be tuned) 
C = 1.

class Block:
    def __init__(self, cols, likelihood_h1=None, likelihood_h2=None, read_indexes=None):
        self.cols = np.array(cols)
        self.likelihood_h1 = likelihood_h1
        self.likelihood_h2 = likelihood_h2
        self._hap = None
        
        if read_indexes is None:
            read_idx = np.zeros(Frags.shape[0], dtype="bool")
            for col in cols:
                read_idx |=  ~np.isnan(Frags[:, col])
            read_indexes = np.nonzero(read_idx)[0]            
        self.read_indexes = read_indexes
        # Reads that contain non nan values at atleast one of the cols.
    
    @property
    def hap(self):
        if self._hap is None:
            self._hap = [H1.take(self.cols), H2.take(self.cols)]
        return self._hap

    @hap.setter
    def hap(self, haplotype):
        self._hap = haplotype
    
    
    
    @property
    def likelihood(self):
        return self.get_likelihood_given_haplotype() 

    def hash(self):
        return hash(tuple(self.cols))
    
    def get_reads(self):
        return Frags[self.read_indexes][:, self.cols]
    
    def get_intersecting_reads(self, other):
        r_idxs = np.intersect1d(self.read_indexes, other.read_indexes)
        cols = np.concatenate([self.cols, other.cols])
        return Frags[r_idxs, :][:, cols]
    
    def is_connected(self, other):
        if len(np.intersect1d(self.read_indexes, other.read_indexes)):
        # if len(set(self.read_indexes).intersection(other.read_indexes)):
            return True
        return False


    def get_likelihood_given_haplotype(self):
        if self.likelihood_h1 is None or self.likelihood_h2 is None:            
            cols = self.cols
            read_indexes = self.read_indexes
            # import ipdb;ipdb.set_trace()
            reads, quals =  Frags.take(read_indexes, axis=0), Quals.take(read_indexes, axis=0)
            assert reads.shape[1] == quals.shape[1] == Frags.shape[1]

            reads, quals = reads.take(cols, axis=1), quals.take(cols, axis=1)
            nan_reads = np.isnan(reads)

            assert reads.shape[1] == len(cols)

            
            hap = self.hap  
            correct_r_h1 = np.array(reads == hap[0]).astype("int32")
            correct_r_h2 = np.array(reads == hap[1]).astype("int32")
            
            like_h1 = (1 - 2 * quals) * correct_r_h1 + quals
            like_h2 = (1 - 2 * quals) * correct_r_h2 + quals

            # TODO: Handle nans carefully, here we are giving the prob that it is 
            # coming from h1, h2 both as 1., maybe it would be suitable to give value as 0.5 
            like_h1[nan_reads] = 1. 
            like_h2[nan_reads] = 1.

            # self.likelihood_h1, self.likelihood_h2 = np.ones(Frags.shape[0]), np.ones(Frags.shape[0])
            self.likelihood_h1, self.likelihood_h2 = np.zeros(Frags.shape[0]), np.zeros(Frags.shape[0])

            # self.likelihood_h1[read_indexes] = like_h1.prod(axis=1)
            # self.likelihood_h2[read_indexes] = like_h2.prod(axis=1)
            self.likelihood_h1[read_indexes] = np.log(like_h1.prod(axis=1))
            self.likelihood_h2[read_indexes] = np.log(like_h2.prod(axis=1))
            

        # likelihoods =  (self.likelihood_h1 + self.likelihood_h2) / 2
        likelihoods = np.logaddexp(self.likelihood_h1, self.likelihood_h2) - np.log(2)
        
        # block_likelihood = np.prod(likelihoods)
        block_likelihood = np.sum(likelihoods)
        return block_likelihood



    def get_likelihood_of_merged(self, other):
        """
        Two blocks can be merged with Haplotypes in same order or the reverse order,
        return the likelihood of the reads for each combination. 
        """
        # Get the value of P( b1 U b2  | h1 h2) / ( P(b1 | h1) * P(b2 | h2)).
        h11, h12 = self.hap
        h21, h22 = other.hap
        assert not len(np.intersect1d(self.cols, other.cols))

        cols = np.concatenate([self.cols, other.cols])
        
        read_indexes = np.unique( np.concatenate([self.read_indexes, other.read_indexes]) )

        hap1 = [np.concatenate([h11, h21]), np.concatenate([h12, h22])]
        hap2 = [np.concatenate([h11, h22]), np.concatenate([h12, h21])]
        hap_phased, hap_unphased = hap1, hap2
        _, __ = self.likelihood, other.likelihood

        # lik_phased_h1 = self.likelihood_h1 * other.likelihood_h1
        # lik_phased_h2 = self.likelihood_h2 * other.likelihood_h2

        # lik_unphased_h1 = self.likelihood_h1 * other.likelihood_h2
        # lik_unphased_h2 = self.likelihood_h2 * other.likelihood_h1
        
        # likelihood_phased = np.prod((lik_phased_h1 + lik_phased_h2)/2)
        # likelihood_unphased = np.prod((lik_unphased_h1 + lik_unphased_h2)/2)
        
        lik_phased_h1 = self.likelihood_h1 + other.likelihood_h1
        lik_phased_h2 = self.likelihood_h2 + other.likelihood_h2
        
        lik_unphased_h1 = self.likelihood_h1 + other.likelihood_h2
        lik_unphased_h2 = self.likelihood_h2 + other.likelihood_h1
        
        likelihood_phased = np.sum(np.logaddexp(lik_phased_h1, lik_phased_h2) - np.log(2))       
        likelihood_unphased = np.sum(np.logaddexp(lik_unphased_h1, lik_unphased_h2) - np.log(2))
        

        if likelihood_phased > likelihood_unphased:
            # the merged block should be in phase
            merged_block = Block(cols, 
            likelihood_h1=lik_phased_h1, 
            likelihood_h2=lik_phased_h2, 
            read_indexes=read_indexes)
            merged_block.hap = hap_phased
        
        else:
            merged_block = Block(cols, 
            likelihood_h1=lik_unphased_h1,
            likelihood_h2=lik_unphased_h2,
            read_indexes=read_indexes)
            merged_block.hap = hap_unphased
        
        return merged_block    




        
        

# IDEA: Store the edges for each block, i.e. the blocks that it is connected
# with, so during the merge, we remove this block and all the edge in adjacent
# blocks, while adding new edges to the new_block. 
# But this would require more memory. (This should be faster as we wont need to 
# traverse to all the blocks to check whether or not it is connected.)

class Edge:
    def __init__(self, u, v, merged):
        # The factor by which the likelihood increases if blocks b1 and 
        # b2 are merged.
        self.u = u
        self.v = v
        self.merged = merged
        # self.lik = self.merged.likelihood / (self.u.likelihood * self.v.likelihood)
        self.lik = self.merged.likelihood - (self.u.likelihood + self.v.likelihood)


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
        # O(logn) lookup
        if block in self.blocks:
            return
        sz = len(self.blocks)
        self.blocks.append(block)
        self.hashd[block] = sz
        self.hashes.add(block.hash())
    
    def remove(self, block):
        # O(logn) lookup
        # O(1) remove.
        index = self.hashd.get(block, None)
        if index is None:
            return  

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


def greedy_selection():
    from tqdm import tqdm
    import pickle
    import ipdb;ipdb.set_trace()
    import os
    
    global block_store, heap
    heap = []
    len_blocks = len(block_store.blocks)
    for i in tqdm(range(len_blocks)):
        for j in range(len_blocks):
            if i == j or not block_store.blocks[i].is_connected(block_store.blocks[j]):
                # There are no reads connecting the two blocks as of now, 
                # Hence no need to check whether the two should be merged. 
                continue
            merged_block = block_store.blocks[i].get_likelihood_of_merged(block_store.blocks[j])
            
            edge = Edge(block_store.blocks[i], block_store.blocks[j], merged_block)
            heap.append(edge)

        
    heapq.heapify(heap)
    import ipdb;ipdb.set_trace()
    
    while heap:
        max_edge = heapq.heappop(heap)
        val, b1, b2, merged = max_edge.lik, max_edge.u, max_edge.v, max_edge.merged
        # Check both b1 and b2 are present in the block_store
        # if not simply continue as this is not a valid edge anymore.
        if b1.hash() not in block_store.hashes or b2.hash() not in block_store.hashes:
            continue

        num_partitions = len(block_store.blocks)
        # Breaking condition, if this does not hold
        # the overall likelihood decreases.
        # if np.log(val) < C*np.log(1 - 1/num_partitions):
        
        # This seems to prevent few of the merges. 
        
        if val < C*(1 - 1/num_partitions):    
            break
        

        merge_blocks(b1, b2, merged)
    import ipdb;ipdb.set_trace()    







def correlation_clustering(threshold=0.):
    from tqdm import tqdm
    from copy import deepcopy
    
    global Blocks

    len_blocks = len(Blocks)
    
    clusters = []
    for i in tqdm(range(len_blocks)):
        new_block = Blocks[i]
        if not len(clusters.blocks):
            clusters.append(new_block)
        else:
            max_lik_cluster = None
            cluster_idx = None
            max_lik = threshold
            for idx, cluster in enumerate(clusters):
                merged = cluster.get_likelihood_merged(new_block)
                if merged.likelihood > max_lik:
                    max_lik = merged.likelihood
                    max_lik_cluster = merged
                    cluster_idx = idx
            if cluster_idx is None:
                # Does not go with any of the clusters, 
                # add a new cluster.
                clusters.append(new_block)
            
            if cluster_idx is not None:
                clusters[cluster_idx] = max_lik_cluster
    
    # No need of Blocks now.
    del Blocks
    cluster_store = BlockStore()
    for block in clusters:
        cluster_store.add(block)
        
    # Initial clustering done. clusters stored in cluster_store
                   
    
    global heap
    heap = []
    
    
    for i in tqdm(range(len(cluster_store.blocks))):
        for j in range(len(cluster_store.blocks)):
            if i == j or not cluster_store[i].is_connected(cluster_store[j]):
                continue
            merged_block = cluster_store[i].get_likelihood_merged(cluster_store[j])
            edge = Edge(cluster_store[i], cluster_store[j], merged_block)
            heap.append(edge)
 
    # We now have a set of clusters, satisfying the property,
    # the block goes into the cluster, iff the likelihood value 
    # of the cluster increases.
    #  
    # We now keep merging the clusters 
    #  - Choose the max value of V = P(C1, C2) / (P(C1)* P(C2)
    #    - If V > 1. (threshold), merge the two clusters. 
    #    - Else, V < 1. (threshold), merge the two clusters and 
    #      also take an extra haplotype, which we believe should 
    #      define a mismapped read.
    
    heapq.heapify(heap)
    while heap:
        max_edge = heapq.heappop(heap)
        val, b1, b2, merged = max_edge.lik, max_edge.u, max_edge.v, max_edge.merged
        # Check both b1 and b2 are present in the block_store
        # if not simply continue as this is not a valid edge anymore.
        if b1.hash() not in block_store.hashes or b2.hash() not in block_store.hashes:
            continue

        num_partitions = len(cluster_store.blocks)
        if val < C*(1 - 1/num_partitions):
            merge_mismapped_blocks(b1, b2, merged)
        else:
            merge_blocks(b1, b2, merged)
    return cluster_store        
                
              
        
def merge_mismapped_blocks(b1, b2, b1b2):
    """Check if adding an H_{n+1} in H [H1, H2, .. H_{n}] makes the likelihood 
    better and by howmuch.

    Args:
        b1 (_type_): _description_
        b2 (_type_): _description_
        b1b2 (_type_): _description_
    """
    # threshold is the minimum likelihood of the read getting sampled
    # from the existing vakues of H. 
    # If all the values are less than threshold try to further improve 
    # the values of H1, H2...Hn
    read_indexes = b1b2.read_indexes
    # tau is specified as the quality of the total read, given
    # all the values of h1, h2,..h_n
    tau = 0.9
    cols = b1b2.cols
    reads, quals = Frags.take(read_indexes, axis=0), Quals.take(read_indexes, axis=0)
    reads, quals = reads.take(cols, axis=1), quals.take(cols, axis=1)
    
    hap_likelihood = []
    qual_reads = np.zeros(read_indexes.shape, dtype="bool") 
    for j, hap in enumerate(b1b2.hap):
        correct_read = np.array(reads == hap).astype("int32")
        # element wise multiplication.
        like_hap = (1 - 2*quals) * correct_read + quals
        
        like_hap_row = like_hap.prod(axis=1)
        like_great_than_thresh = like_hap_row > tau 
        qual_reads |= like_great_than_thresh
        hap_likelihood.append(like_hap)
    
    # Stores the values of h_i which matches most to each read
    h_indexes = np.argmax(np.array(hap_likelihood), axis=0)
    assert reads.shape[0] == h_indexes.shape[0]
    
        
        
        
        
        
        
     
    
    
    
                    





        
                



def read_heter_fragments(fragments_path, vcf_path):
    import allel
    _, Frag, Quals = read_fragments(fragments_path)
    callset = allel.read_vcf(vcf_path)
    filter = callset["variants/FILTER_PASS"]
    ls_01 = np.all(np.equal(callset['calldata/GT'], [0,1]), axis=2).T[0]
    ls_10 = np.all(np.equal(callset['calldata/GT'], [1,0]), axis=2).T[0]
    ls_het = ls_01 | ls_10
    filter = filter & ls_het
    return Frag[:, filter], Quals[:, filter]




def get_blocks_for_reads(fragments_path, vcf_path=None,  greedy=True):
    # Assume all the variant sites to be individual blocks.
    # - two blocks can merge iff 
    #   - the two blocks have a read in common. 
    #   - merging these blocks where the probability of the P( b1 U b2  | h1 h2) / ( P(b1 | h1) * P(b2 | h2))
    import ipdb;ipdb.set_trace()
    
    global Frags, Quals, H, H1, H2, block_store, heap, Blocks


    Frags, Quals = read_heter_fragments(fragments_path, vcf_path)
    Quals = np.power(10, -0.1* Quals)
    H1, H2 = np.ones(Frags.shape[1]), np.zeros(Frags.shape[1])
    H = [H1, H2]

    block_store = BlockStore()   
    num_variants = Frags.shape[1]
    num_variants = num_variants//10
    Blocks = [Block([i]) for i in range(num_variants)]
        
    import ipdb;ipdb.set_trace()
    if greedy:
        for block in Blocks:
            block_store.add(block)
        del Blocks    
        greedy_selection()
    else:
        correlation_clustering()
    import ipdb;ipdb.set_trace() 
    
    # TODO: If two block A is not connected with block B and also 
    # is not connected with block C, choose the block which is 
    # nearer to A, and merge the two blocks to one.
    # As this would not affect the total likelihood, but reduce
    # the number of partitions by 1.  
    
    return block_store     



def merge_blocks(b1, b2, b1b2, debug=False):
    if debug:
        print(f"{b1.cols} {b2.cols} \n {b1b2.get_reads()[:10]} \n {b1b2.hap}")
    # Merge blocks and change the value of the haplotype block.
    # import ipdb;ipdb.set_trace() 
    new_block = b1b2
    global block_store, H, H1, H2, heap
    try:
        block_store.remove(b1)
        block_store.remove(b2)
    except:
        import ipdb;ipdb.set_trace()
        print("Some problem occured while deleting the previous blocks")
        block_store.remove(b1)
        block_store.remove(b2)  
     
    block_store.add(new_block)
    H1[new_block.cols], H2[new_block.cols] = new_block.hap[0], new_block.hap[1]
    H = [H1, H2]
    
    # Go through all the connected blocks of b1 and b2 and
    # add new edges to the heap
    for block in block_store.blocks:
        if (block.is_connected(new_block) and 
            # Check the merges should be performed on non overlapping blocks. 
            not len(np.intersect1d(block.cols, new_block.cols))):
            nb = new_block.get_likelihood_of_merged(block)
            edge = Edge(new_block, block, nb)
            heapq.heappush(heap, edge)
    return        
