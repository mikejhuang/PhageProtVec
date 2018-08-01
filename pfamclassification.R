library(caret)
library(kernlab)
library(e1071)
library(Rtsne)
library(ggplot2)
setwd('/phagegenes/genevec/')

#Load SwissProt 2015 pfam and protvec
pfam <- read.delim("family_classification_metadata.tab") #pfam annotations for SwissProt 2015 dataset
protvec <- read.csv("family_classification_protVec.csv", header=FALSE) #Original protvecs from Asgari for SwissProt 2015 dataset

#Load Cherry's NCBI phage protein functions and protvecs 
pfam<- read.csv("cherryall.csv") #All of Cherry's NCBI data, 120949 phage sequences
protvec <- read.csv("CherryAllProtVecs.csv") #Protvecs for Cherry's NCBI data, with Asgari's embedding


ClassifyQuery <- function(strQuery, pfamColumn, negQuery="", visType='tsne', perplexity=10, savePlot = FALSE){
  posind = grep(strQuery, pfamColumn, ignore.case=TRUE)
  pos <- pfam[posind,]
  pos <- cbind(pos,data.frame(label=rep(1,nrow(pos))))
  View(pos)
  neg <- pfam[-posind,]
  #Negative samples from phage
  if(nchar(negQuery)>0){
    neg <- neg[grep(negQuery,neg$FamilyDescription, ignore.case=TRUE),] }
  neg <- cbind(neg,data.frame(label=rep(0,nrow(neg))))
  dat <- rbind(pos,neg[sample(nrow(neg),size=nrow(pos)),])
  dat <- dat[sample(nrow(dat),size=nrow(dat)),]
  dat$label <- as.factor(dat$label)
  features <- protvec[rownames(dat),]
  pca <- prcomp(features) # principal components analysis using correlation matrix
  features <- cbind(features,pca$x[,1])
  #dat<- dat[!is.na(features$X0),]
  #features <- features[!is.na(features$X0),]
  inTrain <- createDataPartition(y=dat$label, p=0.8, list=FALSE)
  Xtrain <- features[inTrain,]
  Ytrain <- dat$label[inTrain]
  Xtest <- features[-inTrain,]
  Ytest <- dat$label[-inTrain]

  if(visType == 'tsne'){
    tsne <- Rtsne(pca$x[,1:5], dims = 2, perplexity=perplexity, verbose=TRUE, max_iter = 500, check_duplicates=FALSE)
    #tsne <- Rtsne(features, dims = 2, perplexity=5, verbose=TRUE, max_iter = 500, check_duplicates=FALSE)
    tsnedf <- data.frame(tsne$Y)
    colnames(tsnedf) <- c('Xtsne','Ytsne')
    dat <- cbind(dat, tsnedf)
    chart = ggplot(dat,aes(Xtsne, Ytsne)) + geom_point(aes(color=label),alpha=0.5) + ggtitle(paste0("ProtVec tSNE:", strQuery, "vs non-", strQuery, " phage proteins"))
    chart
  }
  if(visType == 'pca'){
    dat<-cbind(dat, pca$x[,1:2])
    chart = ggplot(dat,aes(PC1, PC2)) + geom_point(aes(color=label),alpha=0.5) + ggtitle(paste0("ProtVec tSNE: proteins vs non-Ci phage proteins"))
    chart
  }
  if(savePlot == TRUE){
    ggsave(paste0(strQuery, "vs non-", strQuery, "Proteins Asgari Embedding tsne.png"),plot=chart, width=7, height=5, units="in")
  }
  svm <- train(Xtrain,Ytrain,method="svmRadial")
  Ypred <- predict(svm, pca$x[-inTrain,1:2])
  Ypred <- predict(svm, Xtest)
  confusion <- confusionMatrix(Ypred,Ytest)
  print(confusion)
}


#Plot visualization of top X pfams
topPFamsVis <- function(pfamscol, topn, remove.names=""){

  
  if(length(remove.names)>1){
    for(i in 1:length(remove.names)){
      pfam <- pfam[-grep(remove.names[i],pfamscol,ignore.case=TRUE),]
    }
  }
  else if(nchar(remove.names)>0){
    pfam <- pfam[-grep(remove.names,pfamscol,ignore.case=TRUE),]
  }
  
  toppfams <- names(sort(table(pfamscol),decreasing=TRUE))[1:topn]
  
  posind = pfamscol %in% toppfams
  features = protvec[posind,]
  dat <- pfam[posind,]
  pca <- prcomp(features)
  tsne <- Rtsne(pca$x[,1:3], dims = 2, perplexity=50, verbose=TRUE, max_iter = 500, check_duplicates=FALSE)
  tsnedf <- data.frame(tsne$Y)
  colnames(tsnedf) <- c('Xtsne','Ytsne')
  dat <- cbind(dat, tsnedf)
  dat<- cbind(dat, pca$x[,1:2])
  chart = ggplot(dat,aes(Xtsne, Ytsne)) + geom_point(aes(color=FamilyDescription),alpha=0.5) + ggtitle(paste0("ProtVec tSNE: Top 20 PFam in swissProt"))
  chart
  ggsave(paste0("Top20PFamSwissProtTrainedwith20millionEpochsPhageData - tsne.png"),plot=chart, width=9, height=5, units="in")
  chart = ggplot(dat,aes(PC1, PC2)) + geom_point(aes(color=FamilyDescription),alpha=0.5) + ggtitle(paste0("ProtVec PCA: proteins vs non-Ci phage proteins"))
  chart
}
