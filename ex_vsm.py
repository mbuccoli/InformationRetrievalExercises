import numpy as np
import os
import matplotlib as plt

def generate_exercize():
    files=os.listdir('pages')
    files=[f for f in files if f.endswith('.txt')]
    i=np.random.randint(len(files))
    with open(os.path.join('pages',files[i]),'r') as f:
        content=f.read()
    sentences=content.split('\n')
    D=[s.split('d=')[1] for s in sentences if s.startswith('d')]    
    Q=[s.split('q=')[1] for s in sentences if s.startswith('q')]
    if len(Q)>1:
        q=Q[np.random.randint(len(Q))]
    if len(D)>3:
        D=D[np.random.permutations(len(D))[:3]]
    return q, D    

def propose_exercize():
    q, D=generate_exercize()
    print('Given a query')
    print('\t * %s'%q)
    print('And a collection of documents')
    for d in D:
        print('\t * %s'%d)
    model='freq'
    if model=='freq':
        print('And the term-frequency model')
        print('\t * tf_{i,j}=freq_{i,j}')
    print('Determine the ranking of documents '+\
          'collection with respect to the given query using a'+\
          ' Vector Space Model with the following similarity measures:')
    measures=['Euclidean Distance', 'Cosine Similarity']
    for m in measures:
        print('\t * %s'%m)
    return q, D

def define_vocabulary(D):
    V=[]
    words=[]    
    for d in D:
        words_i=d.lower().split(' ')
        words_i=[w for w in words_i if w!='']
        words.append(words_i)
        V.extend(words_i)
    V=list(np.unique(np.array(V)))
    #V=[v for v in V if v!='']
    print('Dictionary={"%s"}'%'", "'.join(V))
    return V, words

def define_idf(words,V, M,N):
    n=np.zeros(M)
    idf=np.zeros(M)

    print()
    print('Inverse term frequency')
    print('t_i\t\tn_i\tidf_i')    
    for i, v in enumerate(V):
        n[i]=len([w for w in words if np.any(np.array(w)==v) ])    
        idf[i]=np.log2(N/n[i])
        if len(v)<8:
            print('%s\t\t%d\t%.2f'%(v,n[i],idf[i]))
        else:
            print('%s\t%d\t%.2f'%(v,n[i],idf[i]))
    #idf=np.log2(N/n)    
    print('\nLet\'s purge terms with idf=0')
    V=list(np.array(V)[np.where(idf>0)])
    idf=idf[np.where(idf>0)]    
    print('Dictionary={"%s"}'%'", "'.join(V))
    return V, idf             

def define_tf(V, words, query=False):
    M=len(V)
    N=len(words)
    tf=np.zeros((N,M))
    print()
    if query:
        print('Term frequency in the query')
        print('t_i\t\ttf_{i,q}')        
    else:
        print('Term frequency')
        header=['tf_,%d'%(i+1) for i in range(N)]
        print('t_i\t\t%s'%('\t'.join(header)))   
    for i_v, v in enumerate(V):
        row=''
        for i_w, w in enumerate(words):
            tf[i_w,i_v]=np.where(np.array(w)==v)[0].size
            row+='\t'+str(int(tf[i_w,i_v]))#+'\t'
        tab=1-int(len(v)/8)
        ins='\t'*tab
        if not query:
            print('%s%s%s'%(v,ins,row))            
        else:
            print('%s\t%s%d'%(v,ins,tf[0,i_v]))
    return tf

def show_weights(V, weights,weights_q):
    print('\nLet\'s write the documents and the query as vectors')
    for i in range(weights.shape[0]):
        print('d%d = %s'%((i+1), str(np.round(weights[i,:],2))))
    print('q  = %s'%(str(np.round(weights_q,2))))

def rank(SCs):
    idxs=np.flipud(np.argsort(SCs))
    ranking=['d%d'%(i+1) for i in idxs]
    print(' > '.join(ranking))
    
def eucl_similarity(w, w_q):
    print('\nEuclidean Distance as SC')
    N=w.shape[0]
    SCs=np.zeros(N)
    for i in range(N):
        w_d=w[i,:]
        w_diff=w_d-w_q
        w_diff=w_diff[np.abs(w_diff)>0]
        diff=np.sqrt(np.sum(w_diff*w_diff))
        SC=1/(1+diff)
        SCs[i]=SC
        print("d_L2(q,d%d)=%.2f; SC(q,d%d)=%.2f"%(i+1,diff,i+1, SC))
    rank(SCs)

def cos_similarity(w, w_q):
    print('\nCosine Similarity as SC')
    N=w.shape[0]
    SCs=np.zeros(N)
    inv_norm_w_q=1/np.linalg.norm(w_q)
    for i in range(N):
        w_d=w[i,:]
        SC=inv_norm_w_q*np.dot(w_d,w_q)/np.linalg.norm(w_d)
        SCs[i]=SC
        print("SC(q,d%d)=%.2f"%(i+1, SC))
    rank(SCs)    
    
def solve(q, D, clean=False):
    if clean:
        return
    V, words=define_vocabulary(D)
    N=len(D)
    M=len(V)
    V, idf=define_idf(words,V, M,N)
    tf= define_tf(V, words)
        
    words_q=q.lower().split(' ')
    words_q=[w for w in words_q if w!='']
    
    tf_q=define_tf(V,[words_q], query=True)
    weights=tf*idf
    weights_q=tf_q.flatten()*idf
    show_weights(V, weights, weights_q)
    eucl_similarity(weights, weights_q)
    cos_similarity(weights, weights_q)
