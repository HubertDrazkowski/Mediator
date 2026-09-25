# Only base R is required. Preserve the source row ordering and missing values.
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 2) stop('Usage: Rscript export_star.R SOURCE.rda DESTINATION.csv')
scope <- new.env()
load(args[1], envir=scope)
if (!exists('STAR', envir=scope)) stop('The source does not contain STAR')
d <- scope$STAR
required <- c('stark','readk','mathk','read1','math1')
if (!all(required %in% names(d))) stop('Missing required STAR columns')
rows <- data.frame(row_id=rownames(d), action=as.character(d$stark),
                   mediator_value=(d$readk+d$mathk)/2,
                   outcome_value=(d$read1+d$math1)/2,
                   stringsAsFactors=FALSE)
write.csv(rows, file=args[2], row.names=FALSE, na='', fileEncoding='UTF-8')
