#ifndef __UTILS__
#define __UTILS__

class GlobalUUDGenerator
{
public:
	static int generate_id();

private:
	static int next_id;
};

#endif  // __UTILS__
