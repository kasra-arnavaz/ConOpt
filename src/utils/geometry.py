import warp as wp


wp.init()


@wp.func
def normal(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3) -> wp.vec3:
    """Computes the unit vector perpendicular to the plane that goes through p1, p2, and p3"""
    tangent_1 = wp.sub(p2, p1)
    tangent_2 = wp.sub(p3, p1)
    return wp.normalize(wp.cross(tangent_1, tangent_2))


@wp.func
def signed_distance(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, q: wp.vec3) -> float:
    """Computes the signed distance between the plane(p1,p2,p3) and a point q.

    Note:
        If q happens to lie on the plane, the signed distance is zero.
        All the points on one side of the plane have a positive distance,
        and all the points on the other side have negative distance.
    """
    return wp.dot(wp.sub(q, p1), normal(p1, p2, p3))


@wp.func
def point_is_in_tetrahedron(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, p4: wp.vec3, q: wp.vec3) -> bool:
    """Determines if point q lies in the tetrahedron that is formed by p1, p2, p3, p4.

    Returns:
        True, if point lies inside or on the tetrahedron.
        False, if point lies outside the tetrahedron.
    """
    a = signed_distance(p2, p3, p4, q) * signed_distance(p2, p3, p4, p1)
    b = signed_distance(p1, p3, p4, q) * signed_distance(p1, p3, p4, p2)
    c = signed_distance(p1, p2, p4, q) * signed_distance(p1, p2, p4, p3)
    d = signed_distance(p1, p2, p3, q) * signed_distance(p1, p2, p3, p4)
    are_positive = a >= 0 and b >= 0 and c >= 0 and d >= 0
    are_negative = a <= 0 and b <= 0 and c <= 0 and d <= 0
    return are_positive or are_negative


@wp.func
def barycentric_coordinates(p1: wp.vec3, p2: wp.vec3, p3: wp.vec3, p4: wp.vec3, q: wp.vec3, i: int) -> float:
    """Computes the barycentric coordinates of point q w.r.t tetrahedron defined by p1, p2, p3, p4.

    Raises:
        ValueError if point q is outside the tetrahedron.

    Note:
        Barycentric coordinates sum to 1 and are non-negative.
        If all 4 coordinates are non-zero the point is inside the tetrahedron.
        If 3 of the coordintes are non-zero, the point q lies on the face defined by those 3 points.
        If 2 of the coordinates are non-zero, the point q lies on the edge between those those 2 points.
        If 1 of the coordinates is non-zero, the point q is that vertice.
    """
    # if not point_in_tetrahedron(p1, p2, p3, p4, q):
    #     raise RuntimeError("Point q has to be inside or on the tetrahedron defied by p1, p2, p3, p4.")
    if i == 0:
        return signed_distance(p2, p3, p4, q) / signed_distance(p2, p3, p4, p1)
    if i == 1:
        return signed_distance(p1, p3, p4, q) / signed_distance(p1, p3, p4, p2)
    if i == 2:
        return signed_distance(p1, p2, p4, q) / signed_distance(p1, p2, p4, p3)
    if i == 3:
        return signed_distance(p1, p2, p3, q) / signed_distance(p1, p2, p3, p4)
