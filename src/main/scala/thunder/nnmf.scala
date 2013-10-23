/**
 * Created with IntelliJ IDEA.
 * User: freemanj11
 * Date: 10/12/13
 * Time: 1:37 PM
 * To change this template use File | Settings | File Templates.
 */

package thunder

import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.util.Vector
import scala.util.Random
import cern.colt.matrix._
import cern.colt.matrix.linalg._
import cern.jet.math._

object nnmf {

  val factory2D = DoubleFactory2D.dense
  val factory1D = DoubleFactory1D.dense
  val alg = Algebra.DEFAULT

  def parseVector(line: String): (Array[Int],DoubleMatrix1D) = {
    var vec = line.split(' ').drop(3).map(_.toDouble)
    val inds = line.split(' ').take(3).map(_.toDouble.toInt) // xyz coords
    //val mean = vec.sum / vec.length
    //vec = vec.map(x => (x - mean)/(mean + 0.1)) // time series
    return (inds,factory1D.make(vec))
  }

  def randomVector(index: Int, seed1: Int, k: Int) : DoubleMatrix1D ={
    val rand = new Random(index*seed1)
    return factory1D.make(Array.fill(k)(rand.nextDouble))
  }

  def max(num: Double) : Double = {
    return Functions.max(num,0)
  }

  def parseVector2(line: String): DoubleMatrix1D = {
    val vec = line.split(' ').map(_.toDouble)
    return factory1D.make(vec)
  }

  def parseLine(line: String): Array[Double] = {
    val vec = line.split(' ').map(_.toDouble)
    return vec
  }

  def printToImage(rdd: RDD[(Array[Int],Double)], w: Int, h: Int, fileName: String): Unit = {
    // TODO: incorporate different z planes
    val X = rdd.map(_._1(0)).toArray()
    val Y = rdd.map(_._1(1)).toArray()
    val RGB = rdd.map(_._2).collect()
    val img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB)
    val raster = img.getRaster()
    (X,Y,RGB).zipped.foreach{case(x,y,rgb) => raster.setPixel(x-1, y-1, Array(rgb,rgb,rgb))}
    ImageIO.write(img, "png", new File(fileName))
  }

  def outerProd(vec1: DoubleMatrix1D, vec2: DoubleMatrix1D): DoubleMatrix2D = {
    val out = factory2D.make(vec1.size,vec2.size)
    alg.multOuter(vec1,vec2,out)
    return out
  }

  def main(args: Array[String]) {

    val master = args(0)
    val inputFile = args(1)
    val outputFileImg = args(2)
    val k = args(3).toInt
    //val seed1 = args(2).toInt

    System.setProperty("spark.executor.memory", "120g")
    val sc = new SparkContext(master, "nnmf", System.getenv("SPARK_HOME"),
      List("target/scala-2.9.3/thunder_2.9.3-1.0.jar"))

    val data = sc.textFile(inputFile).map(parseVector _).cache()
    val w = data.map{case (k,v) => k(0)}.top(1).take(1)(0)
    val h = data.map{case (k,v) => k(1)}.top(1).take(1)(0)
    val n = data.count().toInt
    val m = data.first()._2.size
    var iter = 0
    val nIter = 10

    // assuming m >> n

    // random initialization
    var u = factory2D.make(k,m)
    val seed1 = Random.nextInt*1000
    var v = data.map{case (k,v) => k(0) + (k(1)-1)*h}.map(randomVector(_,seed1,k))

    println(v.count())
    println(w)
    println(h)

    // fixed initialization
    //var u = factory2D.make(sc.textFile("data/h0.txt").map(parseLine _).toArray())
    //var w = sc.textFile("data/w0.txt").map(parseVector2 _)

    while (iter < nIter) {

      // goal is to solve R = VU subject to U,V > 0
      // by iteratively updating U and V with least squares and clipping

      println("starting" + iter.toString)

      // precompute inv(V' * V)
      val vinv = alg.inverse(v.map( x => outerProd(x,x)).reduce(_.assign(_,Functions.plus)))

      // update U using least squares by premultiplying R component wise with inv(V' * V) * V
      u = data.map(_._2).zip(v.map (x => alg.mult(vinv,x))).map( x => outerProd(x._2,x._1)).reduce(_.assign(_,Functions.plus))

      // clip negative values
      u.assign(Functions.bindArg1(Functions.max,0))

      // precompute pinv(U)
      val uinv = alg.transpose(alg.inverse(alg.transpose(u)))

      // update V using least squares by multiplying R component wise with pinv(U)
      v = data.map(_._2).map( x => alg.mult(alg.transpose(uinv),x))

      // clip negative values
      v = v.map(_.assign(Functions.bindArg1(Functions.max,0)))

      iter += 1

    }

    for (i <- 0 until k) {
      val result = v.map(x => x.get(i))
      val mx = result.top(1).take(1)(0)
      printToImage(data.map(_._1).zip(result).map{case (k,v) => (k,(255*(v/mx)).toInt)}, w, h, outputFileImg + i.toString + ".png")
    }

  }

}
