rm(list=ls())
graphics.off()

require(data.table)
require(ggplot2)
library(grid)
require(gridExtra)


last = 20000

data_pos = fread('log_position')
data_sensors = fread('log_sensors')
data_predictions = fread('log_predictions')
n_pos_rows= dim(data_pos)[1]
if(n_pos_rows <= last)
{
    last = n_pos_rows-1
}
n_sensors_rows= dim(data_sensors)[1]
if(last>0)
    data_pos = data_pos[(n_pos_rows-last):n_pos_rows,]
    data_sensors = data_sensors[(n_sensors_rows-last):n_sensors_rows,]

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


data_sensors = melt(data_sensors, id.vars=c("V21"))
names(data_sensors) <- c('goal', 'sensor', 'activation')

data_predictions$idx = seq(dim(data_predictions)[1])
data_predictions = melt(data_predictions, id.vars=c("V50","idx"))
names(data_predictions) <- c('goal', 'idx',  'gidx', 'prediction')
levels(data_predictions$gidx) = seq(49)
data_predictions$x = (as.numeric(data_predictions$gidx)-1)%%7 
data_predictions$y = as.integer(floor((as.numeric(data_predictions$gidx)-1)/7))
 
alpha_col =  0.01

dev.new()
grobs = list()
i = 1
for( g in seq(0,48) )
{

    cur = subset(data_pos, GOAL==g)


    if (  dim(cur)[1] > 0 )
    {
        p = ggplot(cur, aes(x=x, y=y, color=as.numeric(TRIAL)) )
        p = p + geom_path(size=0.5, alpha=alpha_col)
        p = p + geom_point(size=0.7, alpha=alpha_col)
        p = p + theme_bw()
        p = p + scale_colour_gradientn(colours=rainbow(length(cur$TRIAL)), guide="none")
        p = p + scale_x_continuous(limits=c(-5,5))
        p = p + scale_y_continuous(limits=c(-5,5))
        p = p + coord_fixed(ratio=1)
        
        p = p + theme(
                      axis.ticks = element_blank(),
                      axis.title.x = element_blank(),
                      axis.title.y = element_blank(), 
                      axis.text.x = element_blank(),
                      axis.text.y = element_blank(),
                      panel.border = element_blank() )


        grobs[[i]] = ggplotGrob(p)
    }
    else
    {
        grobs[[i]] = textGrob("")
    }
    i = i + 1
}

g = grid.arrange(grobs=grobs, matrix_layout=matrix(seq(49), 7, 7) )

grid.draw(g)




dev.new()
grobs = list()
i = 1
for( g in seq(0,48) )
{
    cur = subset(data_sensors, goal==g)
    
    if (  dim(cur)[1] > 0 )
    { 
        cur = cur[,.(goal=unique(goal), mn=mean(activation), sd=sd(activation)), by=.(sensor)]

        p = ggplot(cur, aes(x=sensor, y=mn, group=goal))
        p = p + geom_ribbon( aes( ymin=mn-sd, ymax=mn+sd ), fill="#888888" )
        p = p + scale_y_continuous(limits=c(-.5,1))    
        p = p + geom_line()
        p = p + theme_bw()
        p = p + theme(
                      axis.ticks = element_blank(),
                      axis.title.x = element_blank(),
                      axis.title.y = element_blank(), 
                      axis.text.x = element_blank(),
                      axis.text.y = element_blank(),
                      panel.border = element_blank() )
        grobs[[i]] = ggplotGrob(p)
    }
    else
    {
        grobs[[i]] = textGrob("")
    }
    i = i + 1

}

g = grid.arrange(grobs=grobs, matrix_layout=matrix(seq(49), 7, 7) )

grid.draw(g)
