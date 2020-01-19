###(1)Set single feature of negative samples as the minimal value of ones in positive samples
true_negative=read.table("true_negative.txt",sep="\t",quote="",header=T)   ###true negative dataset
positive=read.table("positive.txt",sep="\t",quote="",header=T)  ##positive dataset
temp=true_negative
for(i in 2:(ncol(temp)-2)){
n=which(colnames(positive)==colnames(temp)[i])
positive_min=min(positive[,n])
temp[,i]=positive_min
write.table(temp,paste("single/",colnames(temp)[i],".csv"),sep=",",quote=F,row.names=F)
temp=true_negative
}

###(2)Set feature pairs of negative samples as the minimal value of ones in positive samples
temp=true_negative
for(i in 2:(ncol(temp)-3)){
for(j in (i+1):(ncol(temp)-2)){
n1=which(colnames(positive)==colnames(temp)[i])
positive_min1=min(positive[,n1])
temp[,i]=positive_min1

n2=which(colnames(positive)==colnames(temp)[j])
positive_min2=min(positive[,n2])
temp[,j]=positive_min2

write.table(temp,paste("combine/",colnames(temp)[i],colnames(temp)[j],".csv"),sep=",",quote=F,row.names=F)
temp=true_negative
}
}
