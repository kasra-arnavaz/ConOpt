body_radius=50;
height=40;
finger_length=170;
finger_width=30;
num_fingers=3;
num_teeth_per_finger=3;

starfish(body_radius, height, finger_length, finger_width, num_fingers, num_teeth_per_finger, show_holes=false);

module starfish(body_radius, height, finger_length, finger_width, num_fingers, num_teeth_per_finger, echo_holes=true, show_holes=false, eps=0.5)
{
    translate([0, 0, height/2])
        body(radius=body_radius, width=finger_width, height=height, num_ports=num_fingers);
    angle = 360/num_fingers;
    for (i=[0:num_fingers-1])
    {
        rotate(a=i*angle, v=[0,0,1])
            translate([body_radius, -finger_width/2, 0])
                finger(length=finger_length, width=finger_width, height=height, num_teeth=num_teeth_per_finger);
        holes(body_radius, height, finger_length, finger_width, num_fingers, num_teeth_per_finger, i*angle, echo_holes, show_holes, eps);
    }
}


module holes(body_radius, height, finger_length, finger_width, num_fingers, num_teeth_per_finger, angle, echo_holes, show_holes, eps)
{
    if (echo_holes || show_holes)
    {
        teeth_length = finger_length / (2*num_teeth_per_finger);
        for (j=[1:2:2*num_teeth_per_finger])
        {
            center = [(j+0.5)*teeth_length+body_radius, 0, 0.75*height];
            shift = [0.5*teeth_length-eps,0,0];
            hole_1 = rotate_point(point=center-shift, angle=angle);
            hole_2 = rotate_point(point=center+shift, angle=angle);
            if (echo_holes) {echo(hole_1); echo(hole_2);}
            if (show_holes) {translate(hole_1)sphere(d=5); translate(hole_2)sphere(d=5);}
        }           
    }
}


module body(radius, width, height, num_ports)
{
    angle = 360/num_ports;
    hull()
    {
        for (i=[0:num_ports-1])
        {
            rotate(a=i*angle, v=[0,0,1])
                translate([radius, 0, 0]) cube([10, width, height], center=true);
        }
     }
    
}


module finger(length, width, height, num_teeth)
{
    teeth_length = length / (num_teeth*2);
    cube([length, width, height/2]);
    for (i=[1:2:2*num_teeth])
    {
        translate([i*teeth_length, 0, height/2])
            cube([teeth_length, width, height/2]);
    }
   
}


function rotate_point(point, angle) =
    [
        [cos(angle), -sin(angle), 0],
        [sin(angle),  cos(angle), 0],
        [         0,           0, 1]
   ] * point;