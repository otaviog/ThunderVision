#include "../common.hpp"

UD_NAMESPACE_BEGIN

class Camera;
class Projection;

class Stereo
{
public:
	enum StereoEye
	{
		Eye1, Eye2
	};

	Stereo();
	Stereo(float aperture, float focallength, float eyeseparation);

	virtual void enable() = 0;
	virtual void applyGL(StereoEye eye, const Camera &camera, const Projection &proj) = 0;
	virtual void disable() = 0;

	Stereo& aperture(float value)
	{
		m_aperture = value;
		return *this;
	}
	
	float aperture() const
	{
		return m_aperture;
	}

	Stereo& focallength(float value)
	{
		m_focallength = value;
		return *this;
	}

	float focallength() const
	{
		return m_focallength;
	}

	Stereo& eyeseparation(float value)
	{
		m_eyeseparation = value;
		return *this;
	}

	float eyesperation() const
	{
		return m_eyeseparation;
	}

protected:
	float m_aperture,
		m_focallength,
		m_eyeseparation;
};

class AnaglyphStereo: public Stereo
{
public:
	AnaglyphStereo();
	AnaglyphStereo(float aperture, float focallength, float eyeseparation);

	void enable();
	void applyGL(StereoEye eye, const Camera &camera, const Projection &proj);
	void disable();

private:

};

class SplitStereo: public Stereo
{
public:
	enum Split
	{
		Vertical, Horizontal
	};

	SplitStereo(Split spl);
	SplitStereo(Split spl, float aperture, float focallength, float eyeseparation);

	void enable();
	void applyGL(StereoEye eye, const Camera &camera, const Projection &proj);
	void disable();

private:
	Split m_split;
	int m_savedviewport[4];
};

UD_NAMESPACE_END
