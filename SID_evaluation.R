library(devtools)

load_all('C:\\Users\\flori\\Downloads\\processx')
load_all('C:\\Users\\flori\\Downloads\\pcalg')

load_all('C:\\Users\\flori\\Downloads\\SID')


architecturelist <- c("Fork", "Mediator", "Vstructure", "Diamond", "7ts", "7ts2h")
nbvarlist <- c(3,3,3,4,7,7)


for(k in 1:6){
	architecture <- architecturelist[k]
	nbvar <- nbvarlist[k]
	print(architecture)


	dir <- paste('C:\\Users\\flori\\Desktop\\Stage_2A\\Code\\Results\\tau_max=1\\PCMCI+_1000points\\PCMCI+_Results_', architecture, '\\gpdc', sep="")
	listdir <- list.files(dir)


	gtfile <- paste('C:\\Users\\flori\\Desktop\\Stage_2A\\Code\\Results\\groundtruth\\', architecture, '_groundtruth.csv', sep="")
	gtgraph <- read.csv(gtfile, header=FALSE)
	gtmatrix <- matrix(0,nbvar,nbvar)

	for(i in 1:nrow(gtgraph)){
		gtmatrix[gtgraph[i,1]+1,gtgraph[i,2]+1] <- 1			
	}

	sidlist <- c()


	for(file in listdir){
		if(endsWith(file, '.csv') & file!='Evaluation.csv'){
			graph <- read.csv(paste(dir,'\\',file, sep=""),header= FALSE)
			adjmatrix <- matrix(0,nbvar,nbvar)
			for(i in 1:nrow(graph)){
				adjmatrix[graph[i,1]+1,graph[i,2]+1] <- 1			
			}

			for(i in 1:ncol(adjmatrix)){
				for(j in 1:nrow(adjmatrix)){
					if(adjmatrix[i,j]==1 & adjmatrix[j,i]==1 & i!=j){
						if(adjmatrix[i,j]==gtmatrix[i,j] & adjmatrix[j,i]!=gtmatrix[j,i]){
							adjmatrix[i,j] <- 0
						}
						else if(adjmatrix[j,i]==gtmatrix[j,i] & adjmatrix[i,j]!=gtmatrix[i,j]){
							adjmatrix[j,i] <- 0
						}
					}
				}
			}
			print(adjmatrix)

			sid = tryCatch({
				structIntervDist(gtmatrix,adjmatrix)
				}, error=function(e){
				print("error in SID")}
				)


			sidlist <- c(sidlist, c(file,sid[[1]]))
			sidlist <- c(sidlist, sid[1])
	
		}
	}
	print("list of the SID")
	print(sidlist)

}




