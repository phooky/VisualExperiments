#!/usr/bin/python3
import math
import itertools
import numpy
from numpy.matlib import matrix

# Points are represented as augmented numpy arrays [x,y,1.]
# Segments are represented as tuples (p0, p1, symbol)
# A rule takes a segment and returns an iterator over the
# generated segments

def p(x,y):
    "Convenience method for generating a point"
    return numpy.matrix([x,y,1.])

# A curve is a continuous sequence of line segments tagged with symbols, 
# with rules for decomposing those segments into further segments.
# Its final output is a series of points tagged with symbols indicating the
# symbol associated with the segment the point is at the end of.
class DeterministicCurve:
    def __init__(self,rules):
        self.rules = rules
    def generate(self, pstart, pend, symbol, level = 1):
        if level == 0:
            yield( (pend, symbol) )
        else:
            try:
                for subseg in self.rules[symbol](pstart,pend,symbol):
                    (ss_start, ss_end, ss_sym) = subseg
                    for s in self.generate(ss_start,ss_end,ss_sym,level-1):
                        yield s
            except:
                yield( (pend, symbol) )

# Generate a matrix that maps the segment ((0,0),(0,1)) to (p0,p1)
def map_y_to(p0,p1):
    delta = numpy.subtract(p1,p0)
    a = delta.A[0][1]
    b = -delta.A[0][0]
    return matrix([ [a, -b, p0.A[0][0]], [b, a, p0.A[0][1]], [0, 0, 1] ])


class Pattern:
    def __init__(self, points=[]):
        self.points = points
    def to_rule(self):
        def rule(p0,p1,sym):
            m = map_y_to(p0,p1)
            last = (m*p(0,0).transpose()).transpose()
            for (sp,ssym) in self.points:
                t = (m*sp.transpose()).transpose()
                yield(last,t,ssym)
                last = t
        return rule

pat_koch_snowflake = { 'B': Pattern([(p(0,1./3),'B'), 
    (p((3.**.5)/6,.5), 'B'), 
    (p(0,2./3), 'B'),
    (p(0,1), 'B')]).to_rule() }

pat_koch_modified_A = [
    (p(0,1./3), 'A'), (p((3.**.5)/6,.5), 'B'), (p(0,2./3), 'A'), (p(0,1), 'B')
]
pat_koch_modified_B = [
    (p(0,1./3), 'B'), (p(-(3.**.5)/6,.5), 'A'), (p(0,2./3), 'B'), (p(0,1), 'A')
]
pat_koch_mod = { 'A': Pattern(pat_koch_modified_A).to_rule(),
        'B' : Pattern(pat_koch_modified_B).to_rule() }

d = DeterministicCurve(pat_koch_mod)
#d = DeterministicCurve(pat_koch_snowflake)

import cairo
surface = cairo.SVGSurface("example.svg", 600, 600)
context = cairo.Context(surface)
context.translate(200,100)
context.scale(300, 300)
context.set_line_width(0.002)
context.move_to(0,0)
for (p1,_) in d.generate( p(0,0),p(0,1),'B', 5):
    p1 = p1.A[0]
    context.line_to(p1[0],p1[1])
for (p1,_) in d.generate(p(0,1),p((3.**.5)/2,0.5),'B',5):
    p1 = p1.A[0]
    context.line_to(p1[0],p1[1])
for (p1,_) in d.generate(p((3.**.5)/2,0.5),p(0,0),'B',5):
    p1 = p1.A[0]
    context.line_to(p1[0],p1[1])
context.stroke()
surface.finish()


