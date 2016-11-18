library(bnlearn)


d = empty.graph(c("a","b","c"))

amat(d) = matrix(c(0,1,0,0,0,0,0,0,0), nrow = 3, ncol = 3, byrow = T)


acyclic(d, directed = TRUE, debug = FALSE)

melancon = function(nodes, n)
{
  grp = empty.graph(nodes)
  node = sample(nodes, 2, replace = F)
  mat = amat(grp)
  mat[node[1],node[2]] = 1
  while (n)
  {
    n = n-1
    ij = sample(nodes, 2, replace = FALSE)
    #T1
    if (mat[ij[1],ij[2]] == 1)
    {
      mat[ij[1],ij[2]] = 0
      amat(grp) = mat
    }else{
      if (path(grp,ij[2],ij[1])==FALSE)
      {
        mat[ij[1],ij[2]] = 1
        amat(grp) = mat
      } 
    }
  }
  return(grp)
} 
  
a = melancon(nodes, 10000)
