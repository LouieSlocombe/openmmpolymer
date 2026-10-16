# A showcase render: a polyethylene/polystyrene blend in its
# periodic cell, with the corner nearest the camera cut away so the chains
# can be followed into the bulk.
#
#   vmd -e docs/figures/blend_toc.tcl -args <final.pdb> [blend.tga]
#
# <final.pdb> is what `python docs/figures/blend_toc.py build` prints. Given an
# output file, the scene is rendered with Tachyon at 1600 x 1600 and VMD quits.
# Without one it stays open: turn the view, then run `cut_toward_camera` to
# move the cut to the corner now facing you, and render from
# File > Render (TachyonInternal, or TachyonLOptiXInternal on an NVIDIA GPU).
#
# The shades are those in blend_toc.py; keep the two alike.

if {[llength $argv] < 1} {
    puts "usage: vmd -e blend_toc.tcl -args <final.pdb> \[out.tga\]"
    quit
}
set structure [lindex $argv 0]
set output [lindex $argv 1]

package require pbctools

# Each polymer's darkest and lightest shade, and the colour IDs its chains
# are drawn in. IDs 17-32 are VMD's spare named colours.
set shades(PE) {{0.10 0.33 0.62} {0.55 0.76 0.94}}
set shades(PS) {{0.80 0.38 0.04} {0.99 0.76 0.36}}
set first_id(PE) 17
set first_id(PS) 25
set n_shades 8

foreach polymer {PE PS} {
    lassign $shades($polymer) dark light
    for {set i 0} {$i < $n_shades} {incr i} {
        set f [expr {double($i) / ($n_shades - 1)}]
        set rgb {}
        foreach d $dark l $light {
            lappend rgb [expr {$d + $f * ($l - $d)}]
        }
        color change rgb [expr {$first_id($polymer) + $i}] {*}$rgb
    }
}
color Display Background white

# The stage's coordinates are unwrapped; wrapping every atom into the cell
# makes it read as a clean cube. Bonds are not drawn, so none is guessed.
set molid [mol new $structure type pdb waitfor all autobonds off]
mol delrep 0 $molid
pbc wrap -molid $molid -all

lassign [pbc get -molid $molid] cell
lassign $cell a b c
lassign [measure minmax [atomselect $molid all]] low high
set centre [vecscale 0.5 [vecadd $low $high]]
set half [list [expr {$a / 2.0}] [expr {$b / 2.0}] [expr {$c / 2.0}]]
set corner_low [vecsub $centre $half]

# A view looking down onto one corner, as a crystal is usually drawn.
display resetview
rotate y by -45
rotate x by 35.264

# The cut: the octant whose outer corner faces the camera, which looks down -z.
proc cut_toward_camera {} {
    global molid centre reps_of
    set rotation [lindex [molinfo $molid get rotate_matrix] 0]
    set best -1e9
    foreach sx {-1 1} {
        foreach sy {-1 1} {
            foreach sz {-1 1} {
                set z [lindex [vectrans $rotation [list $sx $sy $sz]] 2]
                if {$z > $best} {
                    set best $z
                    set signs [list $sx $sy $sz]
                }
            }
        }
    }
    set terms {}
    foreach axis {x y z} sign $signs value $centre {
        if {$sign > 0} {set beyond ">"} else {set beyond "<"}
        lappend terms "$axis $beyond $value"
    }
    set cut "not ([join $terms { and }])"
    foreach {rep residue} $reps_of {
        mol modselect $rep $molid "residue $residue and $cut"
    }
}

# One representation per chain, so each gets its own shade.
mol representation VDW 1.0 20
mol material AOChalky
set reps_of {}
set count(PE) 0
set count(PS) 0
set everything [atomselect $molid all]
set residues [lsort -unique -integer [$everything get residue]]
foreach residue $residues {
    set atoms [atomselect $molid "residue $residue"]
    set polymer [lindex [$atoms get resname] 0]
    $atoms delete
    if {![info exists count($polymer)]} {
        puts "skipping residue $residue: no shade for $polymer"
        continue
    }
    set id [expr {$first_id($polymer) + $count($polymer) % $n_shades}]
    incr count($polymer)
    mol color ColorID $id
    mol selection "residue $residue"
    mol addrep $molid
    lappend reps_of [expr {[molinfo $molid get numreps] - 1}] $residue
}
cut_toward_camera

# The cell's twelve edges, thin and grey.
graphics $molid color gray
graphics $molid material AOChalky
foreach {i j k} {0 0 0  1 0 0  0 1 0  0 0 1  1 1 0  1 0 1  0 1 1  1 1 1} {
    lappend corners [vecadd $corner_low [list \
        [expr {$i * $a}] [expr {$j * $b}] [expr {$k * $c}]]]
}
foreach {p q} {0 1  0 2  0 3  1 4  1 5  2 4  2 6  3 5  3 6  4 7  5 7  6 7} {
    graphics $molid cylinder [lindex $corners $p] [lindex $corners $q] \
        radius 0.35 resolution 16 filled yes
}

display projection orthographic
display depthcue off
display rendermode GLSL
display ambientocclusion on
display aoambient 0.85
display aodirect 0.35
display shadows on
axes location off
scale by 0.9

if {$output ne ""} {
    display resize 1600 1600
    render TachyonInternal $output
    quit
}
