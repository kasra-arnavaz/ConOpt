$fn = 30;

Diameter = [35, 32.5, 30, 27.5, 25, 25, 25, 25, 25];
H_cylinder = [20, 20, 20, 20, 20, 20, 15, 15, 15];
h_cone = 7.5;
Num_holes = [5, 5, 5, 4, 4, 4, 3, 3, 3]; // Should be descending
assert(len(Diameter)==len(H_cylinder), "Diameter and H_cylinder must have equal lengths.");
assert(len(Diameter)==len(Num_holes), "Diameter and Num_holes must have equal lengths.");

caterpillar(Diameter, H_cylinder, h_cone);
hole_pos(Diameter, H_cylinder, h_cone, Num_holes);
//hole_tubes(Diameter, H_cylinder, h_cone, Num_holes);
//mold(Diameter, H_cylinder, h_cone);

module hole_pos(Diameter, H_cylinder, h_cone, Num_holes, margin_r=2.5, d_end=10, eps=0.5)
{
    H_block = calc_H_block(H_cylinder, h_cone);
    Z_shift = calc_z_shift(H_block);
    theta = 360/Num_holes[0];
    for (i=[0: len(Diameter)-1])
    {    
        for (j=[0: Num_holes[i]-1])
        {
        r = Diameter[i]/2 - margin_r;
        x = r*cos(2*j*theta);
        y = r*sin(2*j*theta);
        z = h_cone*(Diameter[i]-2*r)/(Diameter[i]-d_end);
        z_1 = h_cone - z + Z_shift[i];
        z_2 = h_cone + H_cylinder[i] + z + Z_shift[i];
        //translate([x, y, z_1]) sphere(d=3);
        //translate([x, y, z_2]) sphere(d=3);
        echo(x, y, z_1+eps);
        echo(x, y, z_2-eps);
        }
    }
}


module hole_tubes(Diameter, H_cylinder, h_cone, Num_holes, margin_r=2.5, d_hole=2)
{
    theta = 360/max(Num_holes);
    H_block = calc_H_block(H_cylinder, h_cone);
    Z_shift = calc_z_shift(H_block);
    for (j=[0: len(Diameter)-1])
    {   r = Diameter[j]/2 - margin_r;
        for (i=[0:Num_holes[j]-1])
        {
            translate([r*cos(2*i*theta), r*sin(2*i*theta), Z_shift[j]])
                cylinder(h=H_cylinder[j]+2*h_cone, d=d_hole);
        }
    }
}

module mold(Diameter, H_cylinder, h_cone, margin_l=2.5, margin_w=10, margin_h=10)
{
    H_block = [for (hc=H_cylinder) hc+2*h_cone];
    h = [for (i=H_block) 1]*H_block;
    w = max(Diameter);
    l = max(Diameter)/2;
    difference()
    {
        translate([l/2,0,h/2])
            cube([l+margin_l, w+margin_w, h+margin_h], center=true);
        caterpillar(Diameter, H_cylinder, h_cone);
        
    }
}
    
module caterpillar(Diameter, H_cylinder, h_cone)
{
    H_block = calc_H_block(H_cylinder, h_cone);
    Z_shift = calc_z_shift(H_block);

        union()
        {   
            for (i=[0: len(Diameter)-1])
            {
                translate([0, 0, Z_shift[i]])
                    block(Diameter[i], H_cylinder[i], h_cone);
            }
        }
}

module block(diameter, h_cylinder, h_cone, d_end=10)
{
    translate([0, 0, h_cone])
        union()
        {
            cylinder(h=h_cylinder, d=diameter);
            translate([0, 0, h_cylinder])
                cylinder(h=h_cone , d1=diameter, d2=d_end);
            translate([0, 0, -h_cone])
                cylinder(h=h_cone , d1=d_end, d2=diameter);
        }
}

function calc_H_block(H_cylinder, h_cone) = [ for (h=H_cylinder) h+2*h_cone];
function calc_z_shift(H_block)
    = concat(0, [ for (i=0, sum=H_block[0]; i<len(H_block);
                i=i+1, sum=sum+(H_block[i]==undef?0:H_block[i])) sum]);