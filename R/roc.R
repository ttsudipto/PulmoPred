fileName <- "/home/sudipto/data/projects/PulmoPred/output/ROC/roc_summary.csv"

data <- read.csv(fileName)
#print(data)
print(ncol(data))
print(nrow(data))
cols_us <- c("SVM (AUC=0.9056)", "RF (AUC=0.9017)", "NB (AUC=0.8851)", 
          "MLP (AUC=0.9111)")
cols_no_us <- c("SVM (AUC=0.9115)", "RF (AUC=0.9023)", "NB (AUC=0.8921)", 
             "MLP (AUC=0.9251)")

#print(as.array(data$SVM.US.AUC...0.9056......0.0107.))
x <- as.array(data$FPR)
svm_us <- as.array(data$SVM.US.AUC...0.9056......0.0107.)
rf_us <- as.array(data$RF.US.AUC...0.9017......0.0094.)
nb_us <- as.array(data$GNB.US.AUC...0.8851......0.0093.)
mlp_us <- as.array(data$MLP.US.AUC...0.9111......0.0111.)
svm_no_us <- as.array(data$SVM.No.US.AUC...0.9115......0.0143.)
rf_no_us <- as.array(data$RF.No.US.AUC...0.9023......0.0259.)
nb_no_us <- as.array(data$GNB.No.US.AUC...0.8921......0.0152.)
mlp_no_us <- as.array(data$MLP.No.US.AUC...0.9251......0.0141.)

lwd <- c(2,1,1,1)
lty <- c(4,5,3,1)
xlab <- "False positive rate"
ylab <- "True positive rate"
#par(mfrow = c(1, 2))

svg("no_us_plot.svg", width=8, height=8)

plot(x, svm_no_us, xlab = xlab, ylab = ylab, type = "l", lwd = lwd[1], lty = lty[1], cex.lab=1.2)
#title("a.", adj=0, line=0.5)
lines(x, rf_no_us, type = "l", lwd = lwd[2], lty = lty[2])
lines(x, nb_no_us, type = "l", lwd = lwd[3], lty = lty[3])
lines(x, mlp_no_us, type = "l", lwd = lwd[4], lty = lty[4])
legend(x = "bottomright", lty = lty, lwd=lwd, cex=0.9, 
       legend=cols_no_us)

dev.off()

svg("us_plot.svg", width=8, height=8)

plot(x, svm_us, xlab = xlab, ylab = ylab, type = "l", lwd = lwd[1], lty = lty[1], cex.lab=1.2)
#title("b.", adj=0, line=0.5)
lines(x, rf_us, type = "l", lwd = lwd[2], lty = lty[2])
lines(x, nb_us, type = "l", lwd = lwd[3], lty = lty[3])
lines(x, mlp_us, type = "l", lwd = lwd[4], lty = lty[4])
legend(x = "bottomright", lty = lty, lwd=lwd, cex=0.9, 
       legend=cols_us)

dev.off()

