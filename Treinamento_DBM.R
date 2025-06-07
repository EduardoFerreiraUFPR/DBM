# Função sigmoide
sigm <- function(x) {
  1 / (1 + exp(-x))
}

# Treinamento do DBM
do.rbm.train1 <- function(rbm1, batch_x, s, numepochs) {
  n <- ncol(batch_x)
  I <- nrow(batch_x)
  x1 <- batch_x

  maximo <- function(y) max(c(0, y))

  if (rbm1$size[3] == 0) {
    if (rbm1$size[2] == 1) {
      medias <- rbm1$W %*% (x1 - rbm1$alpha) + rbm1$B
      y1 <- rnorm(medias, mean = medias, sd = rbm1$Sigma)
      y1 <- sapply(y1, maximo)

      xn <- x1
      yn <- y1

      for (i in 1:rbm1$cd) {
        media <- t(rbm1$W) %*% (yn - rbm1$beta) / rbm1$Sigma^2 + rbm1$A
        xn <- matrix(rbinom(n = media, prob = sigm(media), size = 1), nrow = rbm1$size[1])
        xn2 <- matrix(rbinom(n = media, prob = sigm(media), size = 1), nrow = rbm1$size[1])

        medias <- rbm1$W %*% (xn - rbm1$alpha) + rbm1$B
        yn <- rnorm(medias, mean = medias, sd = rbm1$Sigma)
        yn <- sapply(yn, maximo)
      }

    } else {
      medias <- c(rbm1$W %*% (x1 - rbm1$alpha) + rbm1$B)
      dps <- c(diag(rbm1$Sigma) %*% matrix(1, rbm1$size[2], n))
      y1 <- matrix(rnorm(medias, mean = medias, sd = dps), nrow = rbm1$size[2])
      y1 <- apply(y1, c(1, 2), maximo)

      xn <- x1
      yn <- y1

      for (i in 1:rbm1$cd) {
        medias <- t(diag(rbm1$Sigma^(-2)) %*% rbm1$W) %*% (yn - rbm1$beta) + rbm1$A
        xn <- matrix(rbinom(n = medias, prob = sigm(medias), size = 1), nrow = rbm1$size[1])
        xn2 <- matrix(rbinom(n = medias, prob = sigm(medias), size = 1), nrow = rbm1$size[1])

        medias <- c(rbm1$W %*% (xn - rbm1$alpha) + rbm1$B)
        dps <- c(diag(rbm1$Sigma) %*% matrix(1, rbm1$size[2], n))
        yn <- matrix(rnorm(medias, mean = medias, sd = dps), nrow = rbm1$size[2])
        yn <- apply(yn, c(1, 2), maximo)
      }
    }
  } else {
    z1 <- sigm(rbm1$C)
    medias <- c(rbm1$W %*% (x1 - rbm1$alpha) + 
                as.vector(t(rbm1$V) %*% ((z1 - rbm1$gama) / rbm1$Sigma1)) + rbm1$B)
    y1 <- matrix(rnorm(medias, mean = medias, sd = rbm1$Sigma), nrow = rbm1$size[2])
    y1 <- apply(y1, c(1, 2), maximo)

    medias <- c(rbm1$V %*% (y1 - rbm1$beta) + rbm1$C)
    z1 <- matrix(rnorm(medias, mean = medias, sd = rbm1$Sigma1), nrow = rbm1$size[3])

    xn <- x1
    yn <- y1
    zn <- z1

    for (i in 1:rbm1$cd) {
      media <- t(rbm1$W) %*% (yn - rbm1$beta) + rbm1$A
      xn <- matrix(rbinom(n = media, prob = sigm(media), size = 1), nrow = rbm1$size[1])
      xn2 <- matrix(rbinom(n = media, prob = sigm(media), size = 1), nrow = rbm1$size[1])

      medias <- c(rbm1$W %*% (xn - rbm1$alpha) + 
                  as.vector(t(rbm1$V) %*% ((zn - rbm1$gama) / rbm1$Sigma1)) + rbm1$B)
      yn <- matrix(rnorm(medias, mean = medias, sd = rbm1$Sigma), nrow = rbm1$size[2])
      yn <- apply(yn, c(1, 2), maximo)

      medias <- c(rbm1$V %*% (yn - rbm1$beta) + rbm1$C)
      zn <- matrix(rnorm(medias, mean = medias, sd = rbm1$Sigma1), nrow = rbm1$size[3])
    }
  }

  # Atualização de pesos
  if (rbm1$size[3] == 0) {
    if (rbm1$size[2] == 1) {
      dW <- (((y1 - rbm1$beta) %*% t(x1 - rbm1$alpha)) / rbm1$Sigma^2 -
             ((yn - rbm1$beta) %*% t(xn - rbm1$alpha)) / rbm1$Sigma^2) / n
    } else {
      dW <- (diag(rbm1$Sigma^(-2)) %*% (y1 - rbm1$beta) %*% t(x1 - rbm1$alpha) -
             diag(rbm1$Sigma^(-2)) %*% (yn - rbm1$beta) %*% t(xn - rbm1$alpha)) / n
    }

    dW <- rbm1$learningrate * dW
    rbm1$vW <- rbm1$vW * rbm1$momentum + dW
    rbm1$W <- rbm1$W + rbm1$vW

  } else {
    dW <- (diag(rbm1$Sigma^(-2)) %*% (y1 - rbm1$beta) %*% t(x1 - rbm1$alpha) -
           diag(rbm1$Sigma^(-2)) %*% (yn - rbm1$beta) %*% t(xn - rbm1$alpha)) / n

    dV <- (((z1 - rbm1$gama) / rbm1$Sigma1) %*% t(y1 - rbm1$beta) -
           ((zn - rbm1$gama) / rbm1$Sigma1) %*% t(yn - rbm1$beta)) / n

    dW <- rbm1$learningrate * dW
    dV <- rbm1$learningrate * dV
    rbm1$vW <- rbm1$vW * rbm1$momentum + dW
    rbm1$vV <- rbm1$vV * rbm1$momentum + dV
    rbm1$W <- rbm1$W + rbm1$vW
    rbm1$V <- rbm1$V + rbm1$vV
  }

  # Atualização dos parâmetros A e B
  dA <- rowMeans(x1 - xn)
  dA <- rbm1$learningrate * dA
  rbm1$vA <- rbm1$vA * rbm1$momentum + dA
  rbm1$A <- rbm1$A + rbm1$vA

  if (rbm1$size[2] == 1) {
    rbm1$A <- rbm1$A + rbm1$eta * rowMeans(t(rbm1$W) %*% (y1 - rbm1$beta) / rbm1$Sigma^2)
  } else {
    rbm1$A <- rbm1$A + rbm1$eta * rowMeans(t(diag(rbm1$Sigma^(-2)) %*% rbm1$W) %*% (y1 - rbm1$beta))
  }

  if (rbm1$size[2] == 1) {
    dB <- ((mean(y1) - rbm1$beta) / rbm1$Sigma^2) -
          ((mean(yn) - rbm1$beta) / rbm1$Sigma^2)
  } else {
    dB <- ((rowMeans(y1) - rbm1$beta) %*% diag(rbm1$Sigma^(-2))) -
          ((rowMeans(yn) - rbm1$beta) %*% diag(rbm1$Sigma^(-2)))
  }

  # Continuação da função de atualização de B, C, gama etc. deve ser colocada aqui, se existir.
  
  return(rbm1)
}
