rm(list=ls())
graphics.off()

require(data.table)
require(ggplot2)
library(grid)
require(gridExtra)






data_pos = fread('log_position')
data_pos$V18 = seq(dim(data_pos)[1])
data_pos = melt(data_pos, id.vars=c("V17", "V18"))

names(data_pos) <- c('GOAL','TRIAL', 'POS_IDX', 'DATA')
data_pos$TRIAL = factor(data_pos$TRIAL)
levels(data_pos$POS_IDX) <- factor(seq(16))
data_pos$AXIS = factor((as.numeric(data_pos$POS_IDX)-1) %% 2)
levels(data_pos$AXIS) = c('x','y') 
data_pos$POS = floor((as.numeric(data_pos$POS_IDX)-1) /2)
data_raw = data_pos[,.(GOAL,TRIAL,AXIS,POS,DATA) ] 

data_pos = dcast(data_raw, GOAL+POS+TRIAL~AXIS,  fun.aggregate = mean, value.var="DATA")


dev.new()

cur = subset(data_pos, GOAL==1)

p = ggplot(cur, aes(x=x, y=y))
p = p + geom_path(aes(group=TRIAL), color=alpha("#000000",0.1), size=1.5)
p = p + geom_point(color=alpha("#000000",0.1),size=3)
p = p + theme_bw()
p = p + scale_x_continuous(limits=c(-5,5))
p = p + scale_y_continuous(limits=c(-5,5))
p = p + theme(
              axis.ticks = element_blank(),
              axis.title.x = element_blank(),
              axis.title.y = element_blank(), 
              axis.text.x = element_blank(),
              axis.text.y = element_blank(),
              panel.border = element_blank() )
print(p)
# 
# # grobs = list()
# # i = 1
# # for( g in sort(unique(data_pos$goal)) )
# # {
# #     cur = subset(data_pos, goal==g)
# #     cur_means = subset(data_pos_means, goal==g)
# # 
# #     p = ggplot(cur, aes(x=x, y=y, group=trial))
# #     p = p + geom_path()
# #     p = p + geom_point()
# #     p = p + geom_path(data=cur_means,aes(x=x, y=y), colour="#ff0000" )
# #     p = p + theme_bw()
# #     p = p + theme(
# #                   axis.ticks = element_blank(),
# #                   axis.title.x = element_blank(),
# #                   axis.title.y = element_blank(), 
# #                   axis.text.x = element_blank(),
# #                   axis.text.y = element_blank(),
# #                   panel.border = element_blank() )
# #     grobs[[i]] = ggplotGrob(p)
# #     i = i + 1
# # }
# # 
# # g = grid.arrange(grobs=grobs, matrix_layout=matrix(seq(49), 7, 7) )
# # 
# # grid.newpage()
# # grid.draw(g)
# # 
# 
# 
# 
# data_sensors = fread('log_sensors')
# data = melt(data_sensors, id.vars=c("V21"))
# names(data_sensors) <- c('goal', 'sensor', 'activation')
# 
# dev.new()
# grobs = list()
# i = 1
# for( g in sort(unique(data_sensors$goal)) )
# {
#     cur = subset(data_sensors, goal==g)
#     cur = cur[,.(goal=unique(goal), mn=mean(activation), sd=sd(activation)), by=.(sensor)]
# 
#     p = ggplot(cur, aes(x=sensor, y=mn, group=goal))
#     p = p + geom_ribbon( aes( ymin=mn-sd, ymax=mn+sd ), fill="#888888" )
#     p = p + geom_line()
#     p = p + theme_bw()
#     p = p + theme(
#                   axis.ticks = element_blank(),
#                   axis.title.x = element_blank(),
#                   axis.title.y = element_blank(), 
#                   axis.text.x = element_blank(),
#                   axis.text.y = element_blank(),
#                   panel.border = element_blank() )
#     grobs[[i]] = ggplotGrob(p)
#     i = i + 1
# }
# 
# g = grid.arrange(grobs=grobs, matrix_layout=matrix(seq(49), 7, 7) )
# 
# grid.newpage()
# grid.draw(g)
