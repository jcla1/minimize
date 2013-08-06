package minimize

import "github.com/jcla1/matrix"
import "math"
import "fmt"

var (
	_ = fmt.Println
)

const (
	_MAX   = 20.0
	_RHO   = 0.01
	_SIG   = 0.5
	_INT   = 0.1
	_EXT   = 3.0
	_RATIO = 100.0
)

type CostGradientFunc func(theta *matrix.Matrix) (cost float64, gradients *matrix.Matrix)

func Fmincg(f CostGradientFunc, theta *matrix.Matrix, maxIter int, verbose bool) *matrix.Matrix {
  length := maxIter

  M := 0.0
  i := 0
  red := 1.0
  lsFailed := 0

  f1, df1 := f(theta)
  if length < 0 {
    i += 1
  }

  s := df1.Scale(-1.0)
  d1 := s.Scale(-1.0).Dot(s)
  z1 := red / (1 - d1)

  for float64(i) < math.Abs(float64(length)) {
    if length > 0 {
      i += 1
    }

    theta0 := theta.Copy()
    f0 := f1
    df0 := df1

    theta, _ = theta.Add(s.Scale(z1))
    f2, df2 := f(theta)
    if length < 0 {
      i += 1
    }

    d2 := df2.Dot(s)

    f3 := f1
    d3 := d1
    z3 := -1.0 * z1

    if length > 0 {
      M = _MAX
    } else {
      M = math.Min(_MAX, -1.0 * float64(length - i))
    }

    success := 0
    limit := -1.0;

    var A float64
    var B float64
    var z2 float64

    for {
      for ((f2 > f1+z1*_RHO*d1) || (d2 > -1.0 * _SIG * d1)) && (M > 0) {
        limit = z1

        if f2 > f1 {
          z2 = z3 - (0.5*d3*z3*z3) / (d3*z3+f2-f3)
        } else {
          A = 6 * (f2-f3) / z3 + 3 * (d2+d3)
          B = 3 * (f3-f2) - z3 * (d3 + 2 * d2)
          z2 = (math.Sqrt(B*B-A*d2*z3*z3) - B) / A
        }

        if math.IsNaN(z2) || math.IsInf(z2, 0) {
          z2 = z3 / 2
        }

        z2 = math.Max(math.Min(z2, _INT*z3), (1-_INT)*z3)
        z1 = z1 + z2

        theta, _ = theta.Add(s.Scale(z2))
        f2, df2 = f(theta)

        M -= 1
        if length < 0 {
          i += 1
        }

        d2 = df2.Dot(s)
        z3 = z3 - z2
      }

      if f2 > f1+z1*_RHO*d1 || d2 > -1.0 * _SIG*d1 {
        break // failure
      } else if d2 > _SIG * d1 {
        success = 1
        break // success!
      } else if M == 0 {
        break //failure
      }

      A = 6 * (f2-f3) / z3 + 3 * (d2+d3)
      B = 3 * (f3-f2) - z3 * (d3 + 2 * d2)

      z2 = -1.0 * d2 * z3 * z3 / (B + math.Sqrt(B*B-A*d2*z3*z3))

      if math.IsNaN(z2) || math.IsInf(z2, 0) || z2 < 0 {
        if limit < -0.5 {
          z2 = z1 * (_EXT - 1)
        } else {
          z2 = (limit - z1) / 2.0
        }
      } else if (limit > -0.5) && (z2+z1 > z1*_EXT) {
        z2 = (limit - z1) / 2.0
      } else if (limit < -0.5) && (z2+z1 > z1*_EXT) {
        z2 = z1 * (_EXT - 1.0)
      } else if z2 < (-1.0 * z3 * _INT) {
        z2 = -1.0 * z3 * _INT
      } else if (limit > -0.5) && (z2 < (limit - z1) * (1.0 - _INT)) {
        z2 = (limit - z1) * (1.0 - _INT)
      }

      f3 = f2
      d3 = d2
      z3 = -1.0 * z2
      z1 = z1 + z2

      theta, _ = theta.Add(s.Scale(z2))
      f2, df2 = f(theta)

      M -= 1
      if length < 0 {
        i += 1
      }

      d2 = df2.Dot(s)
    }

    if success == 1 {
      f1 = f2
      if verbose {
        fmt.Printf("Iteration %d | Cost: %0.15f\n", i, f1)
      }

      numerator := (df2.Dot(df2) - df1.Dot(df2)) / df1.Dot(df1)
      s, _ = s.Scale(numerator).Sub(df2)

      tmp := df1
      df1 = df2
      df2 = tmp

      d2 = df1.Dot(s)

      if d2 > 0 {
        s = df1.Scale(-1.0)
        d2 = s.Scale(-1.0).Dot(s)
      }

      //z1 = z1 * math.Min(_RATIO, d1/(d2-math.SmallestNonzeroFloat64))
      z1 = z1 * math.Min(_RATIO, d1/(d2-1e-15))
      d1 = d2
      lsFailed = 0
    } else {
      theta = theta0
      f1 = f0
      df1 = df0

      if lsFailed == 1 || float64(i) > math.Abs(float64(length)) {
        break
      }

      tmp := df1
      df1 = df2
      df2 = tmp
      s = df1.Scale(-1.0)
      z1 = 1.0 / (1.0 - d1)
      lsFailed = 1
    }
  }

	return theta
}

















