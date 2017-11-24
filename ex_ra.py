import numpy as np

alphabet=np.array(['Alpha','Bravo','Charlie','Delta','Echo','Foxtrot','Golf','Hotel','India','Juliett', 'Kilo', 'Lima','Mike','Night', 'Oscar','Papa','Quebec','Romeo','Sierra', 'Tango', 'Uniform', 'Victor', 'Whiskey', 'X-ray', 'Yankee', 'Zulu'])

class Exercise(object):
    def __init__(self,voters=2, candidates=6,topK=3, display=print):
        self.voters=voters
        self.candidates=candidates
        self.topK=topK
        self.A=np.zeros((self.candidates,self.voters))
        self.display=display
        self.generate()
        self.propose()
        

    def generate(self,):
        
        for a in range(self.candidates):
            self.A[a,:]=np.random.randn(self.voters)*1.5+np.random.rand(1)*5+2.5
    
        self.A[self.A<0]=0.05
        self.A[self.A>10]=10
        
    def propose(self):
        self.display('A query is proposed to a generic IR system and the following rates are obtained\n')
        self.display('Result\t\t'+'\t'.join(['Rank%d'%(k+1) for k in range(self.voters)]))
        for i_a in range(self.A.shape[0]):            
            a=alphabet[i_a]
            self.display(a+'\t\t'+'\t'.join(['%.2f'%self.A[i_a,k] for k in range(self.voters)]))
            
        self.display('Compute the top-%d results using the MedRank, the Fagin\'s and the Fagin\'s threshold algorithms'%self.topK)        
    
    
    def sort_A(self):
        S=[]       
        R=[]
        for i in range(self.A.shape[1]):
            idxs=np.flipud(np.argsort(self.A[:,i]))
            R.append(alphabet[idxs])
            S.append(self.A[idxs,i])
        return np.array(S).T, np.array(R).T
        
    def solve(self, debug=False):
        S,R=self.sort_A()
        self.display('First, let us sort the two rankings\n')
        self.display('\t\t'.join(['Rank%d\tScore%d'%(k+1,k+1) for k in range(self.voters)]))
        #self.display('Rank1\tScore1\t\tRank2\tScore2')
        for i_a in range(self.candidates):
            #self.display('%s\t%.2f\t\t%s\t%.2f'%(R[i_a,0],S[i_a,0],R[i_a,1],S[i_a,1]))
            self.display('\t\t'.join(['%s\t%.2f'%(R[i_a,k],S[i_a,k]) for k in range(self.voters)]))
        self.display('\n\n')
        Res,k=self.solve_medrank(S,R,debug)
        self.display('\n\n')
        self.solve_Fagin(S,R,k,debug)
        self.display('\n\n')
        self.solve_Fagin_th(S,R, debug)
        
        
    
    def solve_medrank(self,S,R, debug=False):
        num={}
        voters=S.shape[1]
        candidates=S.shape[0]
        for a in range(candidates):
            num[alphabet[a]]=0
            
        k=0
        K=0
        Res=[]
        self.display('=== MedRank ===')
        while len(Res)<self.topK:
            if k!=0 and debug:
                self.display('List has not been completely filled, another iteration is needed\n')    
            if debug:
                self.display('Rank position: %d'%(k+1))
            for j in range(voters):
                if debug:
                    self.display('Analyzing %s'%R[k,j])
                num[R[k,j]]+=1
    
                if num[R[k,j]]>self.voters/2 and R[k,j] not in Res:
                    if debug:
                        self.display('%s was found in more than half of the ranks: adding in the list'%R[k,j])
                    Res.append(R[k,j])
                    if len(Res)==self.topK:
                        break
            
            k=k+1
        if debug:
            self.display(' = final ranking = ')
        for i in range(self.topK):
            self.display(Res[i])
        
        return Res,k

    def solve_Fagin(self,S,R,k, debug=False):
    
        alph=np.unique(R[:k,:])
        self.display('=== Fagin\'s algorithm ===')
        if debug:
            self.display('From Medrank, we know that we need to consider the first %d positions with sequential access'%k)
    
        #A_alph=np.array([A[np.where(alphabet==a])[0],:] for a in alph])
        #A_alph=np.array([A[np.where(alphabet==a)[0],:] for a in alph])
        mean_alph=np.mean(self.A,axis=1)
        mean_alph=np.array([mean_alph[np.where(alphabet==a)[0][0]] for a in alph])
        
        if debug:
            self.display('Using random access to retrieve the scores')
            self.display('The results are %s'%str(alph))
            self.display('With the following average score')
            for i in range(mean_alph.size):
                self.display('%s\t%.2f'%(alph[i],mean_alph[i]))
            self.display('\n\n = final ranking =')
        idxs=np.flipud(np.argsort(mean_alph))
        
        self.display('Rank\tScore')
        for i in range(self.topK):
            self.display('%s\t%.2f'%(alph[idxs[i]],mean_alph[idxs[i]]))
            
            
    def solve_Fagin_th(self,S,R, debug=False):
        th=15
        res=Results(self.topK)
        seen={}
        self.display('=== Fagin\'s threshold algorithm ===')            
        for k in range(self.A.shape[0]):
            if debug:
                self.display('\n==> Rank position: %d'%(k+1))
            for j in range(self.voters):
                
                a=R[k,j]
                if a in seen:
                    continue
                
                score=np.mean(self.A[np.where(alphabet==a)[0],:])
                seen[a]=score
                if debug:
                    self.display('%s - score %.2f' %(a,score))
                if res.num<self.topK or score>min_score:
                    if debug:                        
                        self.display('Inserting %s'%a)
                    res.put(a,score)
                    min_score=res.min_score()
                if debug:
                    self.display('\nCurrent top-K list')
                    self.display(str(res))
                    self.display()
            th=np.mean(S[k,:])
            if debug:
                self.display('Threshold: %.2f'%th)
            if res.num<self.topK and debug:    
                self.display('List not full')    
            elif min_score<th and debug:
                
                self.display('Min score is lower than the threshold')    
            if res.num>=self.topK and min_score>=th:
                if debug:                
                    self.display('Min score is higher than the threshold and the list was filled: stop\n')
                break        
        
        self.display('Rank\tScore')
        for i in range(self.topK):
            self.display('%s\t%.2f'%(res.elements[i],res.scores[i]))
        

    
class Results(object):
    def __init__(self, max_items=3):
        self.max_items=max_items
        self.elements=[]
        self.scores=[]
        self.num=0
    def put(self, name, score):
        if self.num==0:
            self.elements.append(name)
            self.scores.append(score)
            self.num=1
            return
        new_el=[]
        new_sc=[]
        score_inserted=False
        for i in range(self.num):
            if self.scores[i]<score and not score_inserted:
                new_el.append(name)
                new_sc.append(score)
                score_inserted=True
                
            new_el.append(self.elements[i])
            new_sc.append(self.scores[i])
        if not score_inserted:
            new_el.append(name)
            new_sc.append(score)
                
        if len(new_el)>self.max_items:
            new_el=new_el[:self.max_items]
            new_sc=new_sc[:self.max_items]
        self.elements=new_el
        self.scores=new_sc
        self.num=len(self.elements)
        
    def min_score(self):
        return self.scores[-1]
    def __str__(self):
        return '\n'.join(['%d\t%s\t%.2f'%(k+1, self.elements[k],self.scores[k]) for k in range(self.num) ])

        
if __name__=="__main__":    
    E=Exercise(voters=3, candidates=7, topK=4)
    E.solve(True)
    
    