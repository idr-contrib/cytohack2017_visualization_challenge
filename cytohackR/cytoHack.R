##################################################################################
## code for the mxNetR analysis carried out during the CytoData Hackaction 2017 ##
##################################################################################

# (adopted from:    http://dmlc.ml/rstats/2015/12/08/image-classification-shiny-app-mxnet-r.html
#           and     https://github.com/thirdwing/mxnet_shiny )
#           also find instruction on how to install mxnet there

# dependencies 
library("drat")
drat:::addRepo("dmlc")
library("mxnet")
library("shiny")
library("imager")


# directory where images are stored
plateDir <- c("")

# get file names 
files <- list.files(plateDir)

# create an empty list to store the final results
finalList <- list()


if (!file.exists("synset.txt")) {
    download.file("http://data.dmlc.ml/mxnet/models/imagenet/inception-bn.tar.gz", destfile = "inception-bn.tar.gz")
    untar("inception-bn.tar.gz")
}


# get the pre-trained network
model <<- mx.model.load("./Inception-BN", iteration = 126)


# get class annotations
synsets <<- readLines("synset.txt")

# define function to pre-process data 
preproc.image <- function(im, mean.image) {
    # crop the image
    shape <- dim(im)
    short.edge <- min(shape[1:2])
    xx <- floor((shape[1] - short.edge) / 2)
    yy <- floor((shape[2] - short.edge) / 2) 
    croped <- crop.borders(im, xx, yy)
    # resize to 224 x 224, needed by input of the model.
    resized <- resize(croped, 224, 224)
    # convert to array (x, y, channel)
    arr <- as.array(resized) * 255
    dim(arr) <- c(224, 224, 3)
    # substract the mean
    normed <- arr - 117
    # Reshape to format needed by mxnet (width, height, channel, num)
    dim(normed) <- c(224, 224, 3, 1)
    return(normed)
}


# run over files and get classes 
for (i in files) {
    img <- load.image(paste(plateDir,i,sep="/"))
    normed <- preproc.image(img, mean.img)
    prob <- predict(model, X = normed)
    finalList[[i]] <- cbind(i,t(prob))
}

final_df <- do.call("rbind.data.frame",finalList)

