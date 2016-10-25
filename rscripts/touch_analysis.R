rm(list=ls())
graphics.off()

require(data.table)
require(ggplot2)
library(grid)
require(gtable)
require(gridExtra)


#last = 20000



#---------------------------------------------------------------------------

data_pos = fread('log_position')
# n_pos_rows= dim(data_pos)[1]
# if(n_pos_rows <= last) last = n_pos_rows-1
# if(last>0) data_pos = data_pos[(n_pos_rows-last):n_pos_rows,]

data_pos$idx = seq(dim(data_pos)[1])
data_pos = melt(data_pos, id.vars=c("V17", "V18","idx"))
names(data_pos) <- c('GOAL','TIMESTEP','TRIAL', 'idx', 'DATA')

data_pos$TRIAL = factor(data_pos$TRIAL)
levels(data_pos$idx) <- seq(16)
data_pos$AXIS = factor((as.numeric(data_pos$idx)-1) %% 2)
levels(data_pos$AXIS) = c('x','y') 
data_pos$POS = floor((as.numeric(data_pos$idx)-1) /2)
data_raw = data_pos[,.(GOAL,TIMESTEP,TRIAL,AXIS,POS,DATA) ] 
data_pos = dcast(data_raw, GOAL+POS+TIMESTEP+TRIAL~AXIS,  fun.aggregate = mean, value.var="DATA")


###--------------------------------------------------------------------------------------------------

alpha_col =  .05

gs = list()
i = 1

grad_legend_grob = function(n_cols = 10)
{
    df = data.frame(list(X=seq(1,n_cols,length.out = 20),Y=seq(1,n_cols,length.out = 20)))
    p = ggplot(df,aes(x=X,y=Y,color=X)) + geom_blank()
    p = p + scale_colour_gradientn(colours=topo.colors(length(df$X)))
    p = p + theme_bw()
    p = p + theme(
        axis.ticks = element_blank(),
        axis.title.x = element_blank(),
        axis.title.y = element_blank(), 
        axis.text.x = element_blank(),
        axis.text.y = element_blank(),
        panel.border = element_blank(),
        legend.title=element_blank(), 
        legend.key.width=unit(.1,'npc'), 
        legend.key.height=unit(.16,'npc'),
        legend.position=c(.5,.5)
    )
    return(ggplotGrob(p))
}

g = grobTree(grad_legend_grob( max(as.numeric(data_pos$TRIAL) )))

gs[[i]]  = g
i = i +1
for( g in seq(0,48) )
{
    
    cur = subset(data_pos, GOAL==g)
    
    
    if (  dim(cur)[1] > 0 )
    {
        p = ggplot(cur, aes(x=x, y=y, group=TRIAL, color=as.numeric(TRIAL)) )
        p = p + geom_path(size=0.5, alpha=alpha_col)
        p = p + geom_point(size=0.8, alpha=alpha_col)
        p = p + scale_colour_gradientn(colours=topo.colors(max(as.numeric(data_pos$TRIAL) )), guide="none")
        p = p + scale_x_continuous(limits=c(-4,4))
        p = p + scale_y_continuous(limits=c(-1,4))
        p = p + coord_fixed(ratio=1)
        p = p + theme_bw()
        p = p + theme(
            axis.ticks = element_blank(),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(), 
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            panel.grid = element_blank(),
            plot.margin= unit(c(0.1,0.1,0.1,0), "npc"), 
            panel.border = element_blank() )
        
        gs[[i]] = grobTree(ggplotGrob(p))
    }
    else
    {
        gs[[i]] = grobTree(textGrob(as.character(g)))
    }
    i = i + 1
}

lay = t(matrix(seq(49),7,7)) +1
lay = cbind(rep(1,7),lay)
lay = cbind(rep(1,7),lay)
lay = cbind(rep(1,7),lay)

print(lay)
print(length(gs))
p.pos = grid.arrange(grobs=gs, layout_matrix=lay )

#grid.newpage()
#grid.draw(h)

ggsave(p.pos, file="positions.svg",width=7, height=5, dpi=200)
ggsave(p.pos, file="positions.png",width=7, height=5, dpi=200)


#---------------------------------------------------------------------------

data_pos = fread('log_position')
# n_pos_rows= dim(data_pos)[1]
# if(n_pos_rows <= last) last = n_pos_rows-1
# if(last>0) data_pos = data_pos[(n_pos_rows-last):n_pos_rows,]

data_pos$idx = seq(dim(data_pos)[1])
data_pos = melt(data_pos, id.vars=c("V17", "V18","idx"))
names(data_pos) <- c('GOAL','TIMESTEP','TRIAL', 'idx', 'DATA')

data_pos$TRIAL = factor(data_pos$TRIAL)
levels(data_pos$idx) <- seq(16)
data_pos$AXIS = factor((as.numeric(data_pos$idx)-1) %% 2)
levels(data_pos$AXIS) = c('x','y') 
data_pos$POS = floor((as.numeric(data_pos$idx)-1) /2)
data_raw = data_pos[,.(GOAL,TIMESTEP,TRIAL,AXIS,POS,DATA) ] 
data_pos = dcast(data_raw, GOAL+POS+TIMESTEP+TRIAL~AXIS,  fun.aggregate = mean, value.var="DATA")


###--------------------------------------------------------------------------------------------------

alpha_col =  .003

gs = list()
i = 1
gs[[i]] = grobTree(textGrob(" "))
i = i+1
for( g in seq(0,48) )
{
    
    cur = subset(data_pos, GOAL==g)
    
    
    if (  dim(cur)[1] > 0 )
    {
        p = ggplot(cur, aes(x=x, y=y, group=TRIAL) )
        p = p + geom_path(size=0.5, alpha=alpha_col, color="black")
        p = p + geom_point(size=0.8, alpha=alpha_col, color="black")
        # p = p + scale_colour_gradientn(colours=gray.colors(max(as.numeric(data_pos$TRIAL) )), guide="none")
        p = p + scale_x_continuous(limits=c(-4,4))
        p = p + scale_y_continuous(limits=c(-1,4))
        p = p + coord_fixed(ratio=1)
        p = p + theme_bw()
        p = p + theme(
            axis.ticks = element_blank(),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(), 
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            panel.grid = element_blank(),
            plot.margin= unit(c(0.1,0.1,0.1,0), "npc"), 
            panel.border = element_blank() )
        
        gs[[i]] = grobTree(ggplotGrob(p))
    }
    else
    {
        gs[[i]] = grobTree(textGrob(as.character(g)))
    }
    i = i + 1
}

lay = t(matrix(seq(49),7,7)) +1
lay = cbind(rep(1,7),lay)
lay = cbind(rep(1,7),lay)
lay = cbind(rep(1,7),lay)

print(lay)
print(length(gs))
p.pos = grid.arrange(grobs=gs, layout_matrix=lay )

#grid.newpage()
#grid.draw(h)

ggsave(p.pos, file="positions_greys.svg",width=7, height=5, dpi=200)
ggsave(p.pos, file="positions_greys.png",width=7, height=5, dpi=200)


#---------------------------------------------------------------------------

data_sensors = fread('log_sensors')
data_sensors$idx = seq(dim(data_sensors)[1])
#n_sensors_rows = dim(data_sensors)[1]
#data_sensors = data_sensors[(n_sensors_rows-last):n_sensors_rows,]
data_sensors = melt(data_sensors, id.vars=c("V21","V22","idx"))
names(data_sensors) <- c('goal', 'timestep', 'idx','sensor', 'activation')


#---------------------------------------------------------------------------

data_predictions = fread('log_predictions')
data_predictions$idx = seq(dim(data_predictions)[1])
data_predictions = melt(data_predictions, id.vars=c("V50","V51","idx"))
names(data_predictions) <- c('goal', 'timestep','idx',  'gidx', 'prediction')
levels(data_predictions$gidx) = seq(49)
data_predictions$x = (as.numeric(data_predictions$gidx)-1)%%7 
data_predictions$y = as.integer(floor((as.numeric(data_predictions$gidx)-1)/7))
data_predictions$x_y = paste( 
    as.character(data_predictions$x),
    as.character(data_predictions$y),
    sep="_" )

p.predict = ggplot(data_predictions, aes(x=timestep, 
                                 y=prediction,
                                 group=x_y, color=x_y))
p.predict = p.predict + geom_line(show.legend=FALSE)
p.predict = p.predict + theme_bw()
p.predict = p.predict + theme(
        axis.title.x = element_blank(),
        axis.title.y = element_blank() )

#grid.newpage()
#grid.draw(p)
ggsave(p.predict, file="predictions.svg",width=7, height=5, dpi=200)
ggsave(p.predict, file="predictions.png",width=7, height=5, dpi=200)

idx = data_predictions$idx
timestep = data_predictions$timestep

for(cur_idx in floor( seq(1, max(idx), length.out=50 ) ))
{
    message(cur_idx)
    df = subset(data_predictions, idx==cur_idx)

    dev.new()
    p = ggplot(df, aes(x=x,y=6-y, size=prediction))
    p = p + geom_point(stroke=0)
    p = p + scale_size_continuous(range=c(0,5))

    filename = paste("pred_",sprintf("%012d",timestep[cur_idx]),".png",sep="")
    ggsave(p, file=filename, width=7, height=5, dpi=100)
    dev.off()
}
gc()


#----------------------------------------------------------------

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
        grobs[[i]] = textGrob(as.character(g))
    }
    i = i + 1

}


p.sensors = grid.arrange(grobs=grobs, matrix_layout=matrix(seq(49), 7, 7) )

#grid.draw(g)
ggsave(p.sensors, file="sensors.svg",width=7, height=5, dpi=200)
ggsave(p.sensors, file="sensors.png",width=7, height=5, dpi=200)

ds_mean = data_sensors[,.(mean=mean(activation), sd=sd(activation) ), by=.(sensor)]
p.sensor_general = ggplot(ds_mean, aes(x=sensor, y=mean, group="1")) 
p.sensor_general = p.sensor_general + geom_line(size=2)
p.sensor_general = p.sensor_general + geom_errorbar(aes(ymin=mean, ymax=mean+sd), size=1.5, color="red", width=0.01) 
p.sensor_general = p.sensor_general + ylab("Mean activation")
p.sensor_general = p.sensor_general + xlab("Sensors")
p.sensor_general = p.sensor_general + theme_bw()
p.sensor_general = p.sensor_general + theme(
    panel.border = element_blank() )

#grid.draw(g)
ggsave(p.sensor_general, file="sensors_gen.svg",width=7, height=3, dpi=200)
ggsave(p.sensor_general, file="sensors_gen.png",width=7, height=3, dpi=200)


#----------------------------------------------------------------

data_targets = fread('log_targets')
data_targets$idx = seq(dim(data_targets)[1])
data_targets = melt(data_targets, id.vars=c("V295","V296","idx"))
names(data_targets) <- c('goal', 'timestep','idx',  'target_positions', 'angle')
levels(data_targets$target_positions) = seq(294)
data_targets$target = floor((as.numeric(data_targets$target_positions)-1)/6)
data_targets$position = (as.numeric(data_targets$target_positions)-1)%%6

   
gs = list()
i = 1
for( g in seq(0,48) )
{
    
    cur = subset(data_targets, target==g)
    
    
    if (  sum(! cur$angle %in% NaN ) > 0 )
    {

        p = ggplot(cur, aes(x=position, y=angle, group=target, color=as.numeric(idx)) )
        p = p + geom_path(size=0.5, alpha=alpha_col)
        p = p + geom_point(size=0.8, alpha=alpha_col)
        p = p + scale_colour_gradientn(colours=topo.colors(max(as.numeric(data_targets$TRIAL) )), guide="none")
        p = p + scale_y_continuous(limits=c(-1,4))
        p = p + coord_fixed(ratio=1)
        p = p + theme_bw()
        p = p + theme(
            axis.ticks = element_blank(),
            axis.title.x = element_blank(),
            axis.title.y = element_blank(), 
            axis.text.x = element_blank(),
            axis.text.y = element_blank(),
            panel.grid = element_blank(),
            plot.margin= unit(c(0.1,0.1,0.1,0), "npc"), 
            panel.border = element_blank() )
        

        gs[[i]] = grobTree(ggplotGrob(p))
    }
    else
    {
        gs[[i]] = grobTree(textGrob(as.character(g)))
    }
    i = i + 1
}





ggsave(p.pos, file="positions.svg",width=7, height=5, dpi=200)
ggsave(p.pos, file="positions.png",width=7, height=5, dpi=200)

