#include "quat.hpp"

UD_NAMESPACE_BEGIN

Quatf quatSlerp(const Quatf &lfs, const Quatf &rhs, float t)
{
	// Author : Ketan Mehta
	// Taken from his Quaternion implementation
	// Spherical linear interpolation. Taking two unit quaternions on the unit sphere, slerp
	//      interpolates between the two quats.
	// Implemented as Eq 3.52 in Realtime rendering book.

	const float EPSILON = 0.0001f;
	const float PID2 = UD_FPI / 2.0f;
		float scaleQ, scaleR;
        Quatf q, r;
		q = lfs;
		r = rhs;
        // Ensure t is in the range [0,1]
        if ( t < 0 ) t = 0; 
        else if ( t > 1 ) t = 1;
        
        // Check for unit quaternion - remove it later as it should not have been here..
        if ( q.length() != 1 ) q.normalize();
        if ( r.length() != 1 ) r.normalize();
        
        float cos_theta = quatDot(q, r);  
        float theta = acos(cos_theta);      
        float invSin = 1.0f / sin(theta) ;
        
        // Check for inverting the rotation
        Quatf val;
        
        // Travel along the shorter path. Ref : Adv Anim. by Watt & Watt
        if ( (1.0f + cos_theta) > EPSILON )
        {
                // If angle is not small use SLERP.
                if ( (1.0 - cos_theta) > EPSILON )
                {
					scaleQ = std::sin( (1.0f -t)*theta ) * invSin ;
                        scaleR = std::sin(t*theta) * invSin;
                }
                else    // For small angles use LERP
                {       
//                         std::cout << " are we using LERP " << std::endl;
                        scaleQ = 1.0f - t;
                        scaleR = t;
                }
                val = q * scaleQ  + r * scaleR ;
        }
        else // This is a long way
        {
                // Clear the concept later...
                val.set(-r.w, r.x, -r.y, r.z);
                scaleQ = std::sin( (1.0f - t)*PID2 );
                scaleR = std::sin( t * PID2 );
                val = val*scaleR;
                q = q*scaleQ;
                val = val + val;
        }
        val.normalize();
        return val;
}

Quatf quatEuler(float heading, float attitude, float bank)
{
	// Taken from http://www.euclideanspace.com/maths/geometry/rotations/conversions/eulerToQuaternion/index.htm

	Quatf q;
	
	float		angle;
	float		sr, sp, sy, cr, cp, cy;

	// FIXME: rescale the inputs to 1/2 angle
	angle = bank * 0.5f;
	sy = std::sin(angle);
	cy = std::cos(angle);
	angle = attitude * 0.5f;
	sp = std::sin(angle);
	cp = std::cos(angle);
	angle = heading * 0.5f;
	sr = std::sin(angle);
	cr = std::cos(angle);

	q.x = sr*cp*cy-cr*sp*sy; 
	q.y = cr*sp*cy+sr*cp*sy;
	q.z = cr*cp*sy-sr*sp*cy;
	q.w = cr*cp*cy+sr*sp*sy;

	return q;
}

std::ostream& operator<<(std::ostream &out, const Quatf &v)
{
    out<<"w: "<<v.w<<" "<<v.x<<" "<<v.y<<" "<<v.z;
    return out;
}

UD_NAMESPACE_END
