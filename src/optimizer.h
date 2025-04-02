#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#define EPSILON 1e-8

#include <cmath>
#include <memory>

class Optimizer {
  public:
    double learning_rate;
    Optimizer(double lr) : learning_rate(lr) {}
    virtual ~Optimizer() = default;

    virtual void update(Matrix& param, const Matrix& grad) = 0;
    virtual void update(double& param, double grad) = 0;
    virtual std::unique_ptr<Optimizer> clone() const = 0;
};

class SGD : public Optimizer {
  public:
    SGD(double lr = 0.01) : Optimizer(lr) {}
    
    void update(Matrix& param, const Matrix& grad) override {
        param -= learning_rate * grad;
    }

    void update(double& param, double grad) override {
        param -= learning_rate * grad;
    }

    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<SGD>(learning_rate);
    }
};

class Momentum : public Optimizer {
  private:
    double alpha;
    Matrix velocity_matrix;
    double velocity_scalar;

  public:
    Momentum(double lr = 0.01, double alpha = 0.9) : Optimizer(lr), alpha(alpha), velocity_scalar(0.0) {}

    void update(Matrix& param, const Matrix& grad) override {
        if (velocity_matrix.rows == 0) velocity_matrix = Matrix(param.rows, param.cols);
        velocity_matrix = (alpha * velocity_matrix) - (learning_rate * grad);
        param += velocity_matrix;
    }
    
    void update(double& param, double grad) override {
        velocity_scalar = (alpha * velocity_scalar) - (learning_rate * grad);
        param += velocity_scalar;
    }

    std::unique_ptr<Optimizer> clone() const override {
        return std::make_unique<Momentum>(learning_rate, alpha);
    }
};

class AdaGrad : public Optimizer {
  private:
    Matrix G_matrix;
    double G_scalar;
  
  public:
    AdaGrad(double lr = 0.01) : Optimizer(lr), G_scalar(0.0) {}
    
    void update(Matrix& param, const Matrix& grad) override {
        if (G_matrix.rows == 0) G_matrix = Matrix(param.rows, param.cols);
        G_matrix += grad.hadamard_power(2);
        param -= (learning_rate / (G_matrix.hadamard_power(0.5) + EPSILON)) % grad;
    }

    void update(double& param, double grad) override {
        G_scalar += grad * grad;
        param -= (learning_rate / (std::sqrt(G_scalar) + EPSILON)) * grad;
    }

    std::unique_ptr<Optimizer> clone() const override {
      return std::make_unique<AdaGrad>(learning_rate);
    }
};

class RMSProp : public Optimizer {
};

class Adam : public Optimizer {
};

#endif