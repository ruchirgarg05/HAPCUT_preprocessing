


from utils import *
from sim_util import *
import heapq

# Global Variables
Frags, Quals = None, None
H, H1, H2 = None, None, None
block_store = None
cluster_store = None
heap = None
INF = 4e9

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
    def __init__(self, cols, likelihood_h1=None, likelihood_h2=None, read_indexes=None):
        self.cols = cols
        self.likelihood_h1 = likelihood_h1
        self.likelihood_h2 = likelihood_h2
        
        if read_indexes is None:
            read_idx = np.zeros(Frags.shape[0], dtype="bool")
            for col in cols:
                read_idx |=  ~np.isnan(Frags[:, col])
            read_indexes = np.nonzero(read_idx)[0]            
        self.read_indexes = read_indexes
        # Reads that contain non nan values at atleast one of the cols.
    
    @property
    def hap(self):
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

    def is_connected(self, other):
        if len(set(self.read_indexes).intersection(other.read_indexes)):
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

            self.likelihood_h1, self.likelihood_h2 = np.ones(Frags.shape[0]), np.ones(Frags.shape[0])

            self.likelihood_h1[read_indexes], self.likelihood_h2[read_indexes] = like_h1.prod(axis=1), like_h2.prod(axis=1)

        likelihoods =  (self.likelihood_h1 + self.likelihood_h2) / 2
        likelihoods = np.prod(likelihoods)
        return np.prod(likelihoods)



    def get_likelihood_of_merged(self, other):
        """
        Two blocks can be merged with Haplotypes in same order or the reverse order,
        return the likelihood of the reads for each combination. 
        """
        # Get the value of P( b1 U b2  | h1 h2) / ( P(b1 | h1) * P(b2 | h2)).
        h11, h12 = self.hap
        h21, h22 = other.hap

        cols = list(set(self.cols + other.cols))
        read_indexes = np.unique( np.concatenate([self.read_indexes, other.read_indexes]) )

        hap1 = [np.concatenate([h11, h21]), np.concatenate([h12, h22])]
        hap2 = [np.concatenate([h11, h22]), np.concatenate([h12, h21])]
        hap_phased, hap_unphased = hap1, hap2
        _, __ = self.likelihood, other.likelihood

        lik_phased_h1 = self.likelihood_h1 * other.likelihood_h1
        lik_phased_h2 = self.likelihood_h2 * other.likelihood_h2

        lik_unphased_h1 = self.likelihood_h1 * other.likelihood_h2
        lik_unphased_h2 = self.likelihood_h2 * other.likelihood_h1

        likelihood_phased = np.prod((lik_phased_h1 + lik_phased_h2)/2)
        likelihood_unphased = np.prod((lik_unphased_h1 + lik_unphased_h2)/2)

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
        try:
            self.lik = self.merged.likelihood / (self.u.likelihood * self.v.likelihood)
        except:
            import ipdb;ipdb.set_trace()
        if not self.u.likelihood or not self.v.likelihood:
            import ipdb;ipdb.set_trace()

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
    len_blocks = len_blocks // 10
    for i in tqdm(range(len_blocks)):
        for j in range(len_blocks):
            if i == j or not block_store.blocks[i].is_connected(block_store.blocks[j]):
                # There are no reads connecting the two blocks as of now, 
                # Hence no need to check whether the two should be merged. 
                continue
            merged_block = block_store.blocks[i].get_likelihood_of_merged(block_store.blocks[j])
            # ed_val = merged_block.likelihood / (block_store.blocks[i].likelihood * block_store.blocks[j].likelihood)

            edge = Edge(block_store.blocks[i], block_store.blocks[j], merged_block)
            heap.append(edge)

        
    heapq.heapify(heap)
    ipdb.set_trace()
    
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
        if np.log(val) < C*np.log(1 - 1/num_partitions):
            break

        merge_blocks(b1, b2, merged)

class Cluster(Block):
    """
    Same as Block, but can have more than one haplotype. 
    """
    def converge_with_extra_haps(self):
        # modify self.hap
        pass






class ClusterStore(BlockStore):
    def __init__(self, *args, **kwargs):
        self.threshold = kwargs.pop("threshold", 1.)
        super().__init__(*args, **kwargs)

    def get_max_lik(self, cluster):
        max_lik = -1*INF
        max_cluster_idx = None
        clusters = self.blocks
        max_merged_cluster = None

        for cluster_idx, clust_existing in enumerate(clusters):
            merged_cluster = cluster.get_likelihood_of_merged(clust_existing)
            if merged_cluster.likelihood > max_lik:
                max_lik  = merged_cluster.likelihood
                max_cluster_idx = cluster_idx
                max_merged_cluster = merged_cluster
        # merge the cluster with clusters[max_cluster_idx]
        merge_blocks(cluster, clusters[max_cluster_idx], max_merged_cluster)
                

        return min_cluster_idx, min_distance
    
    def add_cluster(self, cluster):
        # Decide which cluster:Block should be merged
        # with this-cluster.
        # cluster is nothing but a block.
        min_cluster_idx, min_dis = self.get_max_lik(cluster)
        if min_dis > self.threshold:
            self.clusters.append(Cluster([block]))
        else:
            self.clusters[min_cluster_idx].add_block(block)


               

def merge_clusters(c1, c2):
    pass



def correlation_clustering():
    global block_store, cluster_store
    threshold = 1.
    cluster_store = ClusterStore()
    for block in block_store.blocks:
        cluster_store.add_cluster(block)
      

    cluster_heap = []

    for i, cluster_1 in enumerate(cluster_store.blocks):
            for j, cluster_2 in enumerate(cluster_store.blocks):
                if i == j:
                    continue
                # Do a simple merge here without adding any new haplotype.
                merged_cluster = cluster_1.get_likelihood_of_merge(cluster_2)
                edge = Edge(cluster_1, cluster_2, merged_cluster)
                cluster_heap.append(edge)


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

    while len(cluster_store.clusters) > 1:
        # Keep merging until we get a single cluster.
        # Check the clusters in the cluster_store
        # and check if we can merge them.
        max_edge = heapq.heappop(cluster_heap)
        cluster_1, cluster_2, merged_cluster = max_edge.u, max_edge.v, max_edge.merged
        if (cluster_1.hash() not in cluster_store.hashes or 
                cluster_2.hash() not in cluster_store.hashes):
                continue
        # TODO: Add early breaking condition, using the likelihood function.    
        merge_clusters(cluster_1, cluster_2)




        
                








def get_blocks_for_reads(fragments_path, greedy=True):
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
    num_variants = Frags.shape[1]
    blocks = [Block([i]) for i in range(num_variants/10)]
    for block in blocks:
        block_store.add(block)
        
    import ipdb;ipdb.set_trace()
    if greedy:
        greedy_selection()
    else:
        correlation_clustering()
    import ipdb;ipdb.set_trace() 
    
    return block_store     



def merge_blocks(b1, b2, b1b2):
    # Merge blocks and change the value of the haplotype block. 
    new_block = b1b2
    global block_store, H, H1, H2, heap
    try:
        block_store.remove(b1)
        block_store.remove(b2)
    except:
        import ipdb;ipdb.set_trace()
        block_store.remove(b1)
        block_store.remove(b2)  
     
    block_store.add(new_block)
    H1[new_block.cols], H2[new_block.cols] = new_block.hap[0], new_block.hap[1]
    H = [H1, H2]
    
    # Go through all the connected blocks of b1 and b2 and
    # add new edges to the heap
    for block in block_store.blocks:
        if block.is_connected(new_block) and not len(set(block.cols).intersection(new_block.cols)):
            try:
                nb = new_block.get_likelihood_of_merged(block)
            except:
                import ipdb;ipdb.set_trace()
                pass
            edge = Edge(new_block, block, nb)
            heapq.heappush(heap, edge)
    return        
