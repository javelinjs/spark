package com.awkin.rouxlr

import java.util.Random
import java.io._

import scala.math.exp
import Vector._
import spark._
import scala.io._

object HdfsLR {
  var D = 1  // Numer of dimensions
  val rand = new Random(42)

  case class DataPoint(x: Vector, y: Double)

  def parsePoint(line: String): DataPoint = {
    //val nums = line.split(' ').map(_.toDouble)
    //return DataPoint(new Vector(nums.slice(1, D+1)), nums(0))
    val tok = new java.util.StringTokenizer(line, " ")
    var y = tok.nextToken.toDouble
    var x = new Array[Double](D)
    var i = 0
    while (i < D) {
      x(i) = tok.nextToken.toDouble; i += 1
    }
    return DataPoint(new Vector(x), y)
  }

  def outputResult(w: Vector, file: String) {
    val out = new PrintWriter(file)
    try { 
        out.println(w) 
    } finally { 
        out.close 
    }
  }

  def main(args: Array[String]) {
    if (args.length < 7) {
      System.err.println("Usage: HdfsLR <master> <file> <dimension> <alpha> <threshold> <max_iteration> <output>")
      System.exit(1)
    }
    val sc = new SparkContext(args(0), "HdfsLR")
    val lines = sc.textFile(args(1))
    D = args(2).toInt
    val points = lines.map(parsePoint _).cache()
    val ALPHA = args(3).toDouble
    val THRESHOLD = args(4).toDouble
    val MAX_ITERATION = args(5).toInt
    val outputFile = args(6)

    // Initialize w to a random value
    var w = Vector(D, _ => 2 * rand.nextDouble - 1)
    var _w = w
    var iter = 0 
    println("Initial w: " + w)

    do {
      val gradient = points.map { p =>
        (1 / (1 + exp(-p.y * (w dot p.x))) - 1) * p.y * p.x
      }.reduce(_ + _)
      _w = w
      w -= ALPHA * gradient
      iter = iter + 1
    } while ((w squaredDist _w) > THRESHOLD && iter < MAX_ITERATION)

    println("Final w: " + w)
    outputResult(w, outputFile)
  }
}
