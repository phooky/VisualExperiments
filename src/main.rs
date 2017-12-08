extern crate rand;
extern crate svg;
extern crate clap;

use svg::Document;
use svg::node::element::Path;
use svg::node::element::path::Data;

#[derive(Clone,PartialEq,Debug)]
struct Point {
    x : f32,
    y : f32
}

#[derive(Clone,PartialEq,Debug)]
struct Segment {
    from : Point,
    to : Point,
}

#[derive(Clone)]
struct Node {
    pos : Point,
    edges : Vec<usize>,
}

#[derive(Clone)]
struct Graph {
    nodes : Vec<Node>
}

struct NodeSet {
    inset : Vec<bool>,
    nodes : Vec<usize>,
}

impl NodeSet {
    fn new(g : &Graph) -> NodeSet {
        NodeSet { inset : vec![false; g.nodes.len()], nodes : Vec::new() }
    }
    fn add(&mut self, i : usize) {
        if !self.inset[i] { self.nodes.push(i); } 
        self.inset[i] = true;
    }
    fn remove(&mut self, i : usize) {
        if self.inset[i] {
            for ridx in 0..self.nodes.len() {
                if self.nodes[ridx] == i {
                    self.nodes.swap_remove(ridx);
                    break
                }
            }
        }
        self.inset[i] = false;
    }
    fn is_empty(&self) -> bool {
        self.nodes.len() == 0
    }
    fn is_full(&self) -> bool {
        self.inset.len() == self.nodes.len() 
    }
    fn size(&self) -> usize {
        self.nodes.len()
    }
    fn has(&self, i : usize) -> bool {
        self.inset[i]
    }
}

impl Point {
    fn normalize(&self) -> Option<Point> {
        if (self.x == 0.0) && (self.y == 0.0) {
            None
        } else {
            let scale = (self.x*self.x + self.y*self.y).sqrt();
            Some( Point { x : self.x/scale, y : self.y/scale } )
        }
    }

    fn tuple(&self) -> (f32, f32) {
        (self.x, self.y)
    }
}
        
impl Segment {
    fn normal(&self) -> Option<Point> {
        let dx = self.to.x - self.from.x; let dy = self.to.y - self.from.y;
        Point{ x : dy, y : -dx }.normalize()
    }
} 

impl Graph {
    fn add_directed_edge(&mut self, from : usize, to : usize) {
        self.nodes[from].edges.push(to)
    }

    fn add_undirected_edge(&mut self, v1 : usize, v2 : usize) {
        self.add_directed_edge(v1, v2);
        self.add_directed_edge(v2, v1);
    }

    fn pos(&self, v : usize) -> Point {
        self.nodes[v].pos.clone()
    }

    fn find_leaf(&self) -> usize {
        let mut i = 0;
        let mut ns = NodeSet::new(self);
        ns.add(i);
        while self.nodes[i].edges.len() > 1 {
            let cnt = self.nodes[i].edges.len();
            for j in 0..cnt {
                let cand = self.nodes[i].edges[j];
                if !ns.has(cand) {
                    ns.add(cand);
                    i = cand;
                    break;
                }
            }
        }
        i
    }

    fn new_square_graph(dim : i32, scale : f32) -> Graph {
        let w = dim as usize;
        let h = dim as usize;
        let mut g = Graph { nodes : Vec::new() };
        for x in 0..w { 
            for y in 0..h { 
                g.nodes.push(Node{ 
                    pos : Point { 
                        x : x as f32 * scale,
                        y : y as f32 * scale,
                    },
                    edges : Vec::new() });
            }
        }
        for x in 0..w {
            for y in 0..h {
                let idx = (y*w)+x;
                if x > 0 {
                    g.add_undirected_edge(idx-1, idx);
                }
                if y > 0 {
                    g.add_undirected_edge(idx-w, idx);
                }
            }
        }
        g
    }
    
    fn new_hex_graph(r : i32, scale : f32) -> Graph {
        let mut g = Graph { nodes : Vec::new() };
        let mut hex_map = std::collections::HashMap::new();
        let s3o2 = (3 as f32).sqrt()/2.0;
        let offsets = vec![ (-1,0), (0, -1), (1, -1) ];
        for y in -r..r+1 {
            for x in -r..r+1 {
                let z : i32 = 0 - (x+y);
                if z < -r || z > r { continue; }
                let hx = (x as f32 * scale) + (y as f32 * scale * 0.5);
                let hy = y as f32 * scale * s3o2;
                let node = Node { pos : Point { x: hx, y: hy }, edges: Vec::new() };
                g.nodes.push(node);
                let nidx = g.nodes.len() -1;
                hex_map.insert( (x,y), nidx );
                // link edges
                for &(offx, offy) in &offsets {
                    match hex_map.get( &(x+offx, y+offy) ) {
                        Some(idx) => g.add_undirected_edge(nidx, *idx),
                        None => (),
                    }
                }
            }
        }
        g
    }
        

    fn make_disconnected_copy(&self) -> Graph {
        let mut g = Graph { nodes : Vec::new() };
        for n in self.nodes.iter() {
            g.nodes.push(Node { pos : n.pos.clone(), edges : Vec::new() });
        }
        g
    }

    fn make_random_tree(&self) -> Graph {
        let mut g = self.make_disconnected_copy();
        let root = rand::random::<usize>() % self.nodes.len();
        let mut ns = NodeSet::new(&g); // Nodes in tree
        let mut ncs = NodeSet::new(&g); // Nodes with available edges to grow
        ns.add(root);
        ncs.add(root);
        let mut rng = rand::thread_rng();
        while !ns.is_full() {
            // select random element from candidates
            let from = *(rand::sample(&mut rng, ncs.nodes.iter(), 1)[0]);
            let mut tos = self.nodes[from].edges.clone();
            while tos.len() > 0 {
                let to = *(rand::sample(&mut rng, tos.iter(), 1)[0]);
                if !ns.has(to) {
                    g.add_undirected_edge(from,to);
                    ns.add(to);
                    ncs.add(to);
                    break
                }
                for i in 0..tos.len() {
                    if tos[i] == to {
                        tos.swap_remove(i);
                        break
                    }
                }
            }
            if tos.len() == 0 {
                ncs.remove(from)
            }
        }
        g
    }

    // Indicate what a "right turn" would be after going to the next
    // node; a dead end yields a reflection
    fn right_turn(&self, from : usize, to : usize) -> usize {
        let to_node = &self.nodes[to];
        if to_node.edges.len() == 1 { return to_node.edges[0]; }
        // create an ordered list of angle/node tuples
        let mut tup_list = Vec::new();
        for n in to_node.edges.iter() {
            let npos = self.pos(*n);
            let dx = npos.x - to_node.pos.x;
            let dy = npos.y - to_node.pos.y;
            let dir = dy.atan2(dx);
            tup_list.push( (dir, *n) )
        }
        tup_list.sort_by(|a,b| a.0.partial_cmp(&b.0).unwrap());
        for i in 0..tup_list.len() {
            if tup_list[i].1 == from {
                return tup_list[ (i+1) % tup_list.len() ].1;
            }
        }
        panic!("Couldn't find incoming node edge! (Directed graph???)");
    }

    fn count_edges(&self) -> usize {
        let mut count = 0;
        for n in self.nodes.iter() {
            count += n.edges.len();
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_square_graph() {
        assert_eq!(Graph::new_square_graph(5,1.0).count_edges(),80);
        assert_eq!(Graph::new_square_graph(8,1.0).count_edges(),224);
    }
    #[test]
    fn test_hex_graph() {
        let g1 = Graph::new_hex_graph(1,1.0);
        assert_eq!(g1.nodes.len(), 7);
        assert_eq!(g1.count_edges(), 24);
        let g = Graph::new_hex_graph(2,1.0);
        assert_eq!(g.nodes.len(), 19);
        assert_eq!(g.count_edges(), 14*3*2);
    }
    #[test]
    fn test_find_leaf() {
        let g = Graph::new_hex_graph(5,1.0).make_random_tree();
        assert_eq!(g.nodes[g.find_leaf()].edges.len(),1);
    }
    #[test]
    fn test_disconnected_copy() {
        let g = Graph::new_square_graph(4,1.0);
        let gd = g.make_disconnected_copy();
        assert_eq!(gd.nodes.len(), g.nodes.len());
        assert_eq!(gd.count_edges(), 0);
    }
    #[test]
    fn test_random_tree() {
        let g = Graph::new_square_graph(20,1.0);
        let rt = g.make_random_tree();
        assert_eq!(rt.count_edges(), (g.nodes.len()-1)*2);
    }
    #[test]
    fn test_simple_normal() {
        let s1 = Segment { 
            from : Point { x : 0.0, y : 0.0 },
            to : Point { x : 0.0, y : 10.0 } 
        };
        let r1 = s1.normal();
        assert_eq!(r1, Some( Point { x : 1.0, y : 0.0 } ) );
        let s2 = Segment {
            from : Point { x : 11.0, y : 11.0},
            to : Point { x : 11.0, y : 11.0 } 
        };
        let r2 = s2.normal();
        assert_eq!(r2, None);
    }
}

use clap::{Arg, App, SubCommand};

fn offset( point : Point, normal : Point, scalar : f32 ) -> Point {
    Point { x : point.x + normal.x * scalar, y : point.y + normal.y * scalar }
}

fn make_path_from_graph(g : &Graph, off : f32) -> Data {
    let root = g.find_leaf();
    let mut data = Data::new();
    let mut last_node = root;
    let mut cur_node = g.right_turn(root, root);
    data = data.move_to(g.pos(cur_node).tuple());
    while cur_node != root {
        let next_node = g.right_turn(last_node,cur_node);
        let seg = Segment { from : g.pos(next_node), to : g.pos(cur_node) };
        match seg.normal() {
            None => (),
            Some( n ) => {
                data = data.line_to(offset(g.pos(next_node),n,off).tuple());
            }
        }
        last_node = cur_node; cur_node = next_node;
    }
    data
}

fn main() {
    let matches = App::new("Graph tree generator")
        .arg(Arg::with_name("topology")
             .short("t")
             .long("topo")
             .takes_value(true)
             .help("\"square\" or \"hex\""))
        .arg(Arg::with_name("scale")
             .short("s")
             .long("scale")
             .takes_value(true))
        .arg(Arg::with_name("dimension")
             .short("d")
             .long("dim")
             .takes_value(true))
        .get_matches();

    let scale = match matches.value_of("scale") {
        Some(x) => x.parse::<f32>().unwrap(),
        None => 10.0,
    };
    let dimension = match matches.value_of("dimension") {
        Some(x) => x.parse::<i32>().unwrap(),
        None => 20,
    };
    let topo = matches.value_of("topology").unwrap_or("hex");
    let g = match topo {
        "square" => Graph::new_square_graph(dimension,scale),
        "hex" => Graph::new_hex_graph(dimension, scale),
        _ => panic!("Unrecognized topology!")
    };
    let tree = g.make_random_tree();

    let mut data = Data::new();
    /*
    for i in 0..tree.nodes.len() {
        let ref n = tree.nodes[i];
        for to in &n.edges {
            if *to > i {
                let ref ton = tree.nodes[*to];
                data = data.move_to((n.x,n.y));
                data = data.line_to((ton.x,ton.y));
            }
        }
    }*/

    data = make_path_from_graph(&tree,3.0);
    let path = Path::new()
        .set("fill", "none")
        .set("stroke", "black")
        .set("stroke-width", 3)
        .set("d", data);

    let document = Document::new()
        .set("viewBox", (-300, -300, 600, 600))
        .add(path);

    svg::save("image.svg", &document).unwrap();
}
