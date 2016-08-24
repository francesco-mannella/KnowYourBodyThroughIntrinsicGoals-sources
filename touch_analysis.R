require(plyr)
require(reshape2)
require(ggplot2)

data = read.table("curr_log")
n_sensors = dim(data)[2]-1

labs= sapply(format(seq(n_sensors), trim=TRUE),FUN=function(x) { paste('p',x,sep="") })
names(data) = c(labs,'idx')

data = melt(data, id.vars="idx")

means = ddply(data, .(idx,variable), summarize, x = mean(value))
pmeans = ddply(means,.(idx), transform, q = x/norm(as.matrix(x)))

qq = matrix(pmeans$q, 7*20, 7 )
image(qq)

dev.new()
gp = ggplot(pmeans, aes(x=variable,y=q)) + facet_wrap(~idx) + geom_point()
print(gp)
